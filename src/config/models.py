"""Dataclass model definitions for orchestrator configuration."""

from __future__ import annotations

import json
import logging
import os
import socket
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.roles import Role
from src.registry.kernel_paths import (
    backend_dir as _kernel_backend_dir,
    server_binary as _kernel_server_binary,
)
from src.registry.stack_priors import live_stack_serving_url_values

from .validation import _registry_runtime_value, _registry_timeout

_LOGGER = logging.getLogger(__name__)

# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class LLMConfig:
    """Configuration for LLM primitives."""

    output_cap: int = 8192
    """Maximum characters per sub-LM output."""

    batch_parallelism: int = 4
    """Maximum parallel calls in llm_batch."""

    call_timeout: int = 600  # Increased from 300 - architect calls can take ~300s
    """Timeout per call in seconds (matches LlamaServerBackend)."""

    mock_response_prefix: str = "[MOCK]"
    """Prefix for mock responses."""

    max_recursion_depth: int = 5
    """Maximum nesting depth for sub-LM calls."""

    default_prompt_rate: float = 0.50
    """Default cost rate per 1M prompt tokens."""

    default_completion_rate: float = 1.50
    """Default cost rate per 1M completion tokens."""

    qwen_stop_token: str = "<|im_end|>"
    """Qwen chat-template stop token to prevent runaway generation."""

    depth_role_overrides: str = field(
        default_factory=lambda: str(
            _registry_runtime_value(("llm", "depth_role_overrides"), "1:worker_general")
        )
    )
    """Optional depth->role override map (CSV or JSON), e.g. "1:worker_general,2:worker_math"."""

    depth_override_max_depth: int = field(
        default_factory=lambda: int(_registry_runtime_value(("llm", "depth_override_max_depth"), 3))
    )
    """Maximum nested depth eligible for override routing."""


@dataclass
class EscalationConfigData:
    """Configuration for escalation policy."""

    max_retries: int = 2
    """Maximum retries before escalation."""

    max_escalations: int = 2
    """Maximum escalations per task."""

    optional_gates: frozenset[str] = field(
        default_factory=lambda: frozenset({"typecheck", "integration", "shellcheck"})
    )
    """Gates that can be skipped on timeout."""


@dataclass
class REPLConfigData:
    """Configuration for REPL environment."""

    max_output_len: int = 10000
    """Maximum output length per execution."""

    timeout_seconds: int = 30
    """Execution timeout."""

    forbidden_modules: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "os",
                "sys",
                "subprocess",
                "shutil",
                "pathlib",
                "socket",
                "http",
                "urllib",
                "ftplib",
                "smtplib",
                "pickle",
                "marshal",
                "shelve",
                "dbm",
                "ctypes",
                "multiprocessing",
                "threading",
                "importlib",
                "builtins",
                "__builtins__",
                "code",
                "codeop",
                "compile",
                "exec",
                "eval",
            }
        )
    )
    """Modules blocked from import."""

    forbidden_builtins: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "__import__",
                "eval",
                "exec",
                "compile",
                "open",
                "input",
                "breakpoint",
                "globals",
                "locals",
                "vars",
                "getattr",
                "setattr",
                "delattr",
                "hasattr",
                "type",
                "object",
                "__build_class__",
                "memoryview",
                "bytearray",
            }
        )
    )
    """Builtins blocked from use."""


@dataclass
class ServerConfigData:
    """Configuration for backend servers."""

    default_url: str = "http://localhost:8080"
    """Default server URL."""

    timeout: int = 600
    """Request timeout in seconds (increased for architect models)."""

    num_slots: int = 2
    """Number of parallel slots (must match llama-server -np value)."""

    connect_timeout: int = 5
    """Connection timeout."""

    retry_count: int = 3
    """Number of retries on failure."""

    retry_backoff: float = 0.5
    """Backoff factor for retries."""


@dataclass
class MonitorConfigData:
    """Configuration for generation monitoring.

    Base defaults used by MonitorConfig(). Per-tier and per-task overrides
    are in tier_overrides and task_overrides, consumed by for_tier()/for_task().
    """

    entropy_threshold: float = 4.0
    """Sustained entropy above this triggers abort."""

    entropy_spike_threshold: float = 2.0
    """Single-token entropy jump threshold."""

    repetition_threshold: float = 0.3
    """Threshold for repeated n-gram ratio (0-1)."""

    min_tokens_before_abort: int = 50
    """Minimum tokens before allowing abort."""

    perplexity_window: int = 20
    """Rolling window size for perplexity trend."""

    max_length_multiplier: float = 2.0
    """Abort if >N x median task length."""

    entropy_sustained_count: int = 10
    """Tokens of high entropy before abort."""

    ngram_size: int = 3
    """N-gram size for repetition detection."""

    combined_threshold: float = 0.7
    """Weighted score for combined signals."""

    tier_overrides: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "worker": {
                "entropy_threshold": 4.5,
                "entropy_spike_threshold": 2.5,
                "min_tokens_before_abort": 50,
            },
            "coder": {
                "entropy_threshold": 5.0,
                "entropy_spike_threshold": 3.0,
                "min_tokens_before_abort": 100,
                "repetition_threshold": 0.2,
            },
            "architect": {
                "entropy_threshold": 6.0,
                "entropy_spike_threshold": 4.0,
                "min_tokens_before_abort": 200,
                "repetition_threshold": 0.4,
            },
            "ingest": {
                "entropy_threshold": 5.5,
                "entropy_spike_threshold": 3.5,
                "min_tokens_before_abort": 100,
            },
        }
    )
    """Per-tier threshold overrides. Keys are tier names, values are dicts of field→value."""

    task_overrides: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "code": {"min_tokens_before_abort": 100, "repetition_threshold": 0.2, "ngram_size": 4},
            "reasoning": {
                "entropy_threshold": 4.5,
                "min_tokens_before_abort": 30,
                "perplexity_window": 15,
            },
        }
    )
    """Per-task threshold overrides. Keys are task types, values are dicts of field→value."""


def _get_default_llm_root() -> str:
    """Get LLM root from environment or default."""
    return os.environ.get("ORCHESTRATOR_PATHS_LLM_ROOT", "/mnt/raid0/llm")


# The checkout this module was loaded from. Derived from __file__ rather than
# from llm_root so a git worktree reads its OWN registry/priors/config instead
# of the main checkout's — control-plane reads must follow the code, not the host.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _get_default_project_root() -> str:
    """Get project root from environment, else the checkout this module lives in."""
    return os.environ.get("ORCHESTRATOR_PATHS_PROJECT_ROOT", str(_REPO_ROOT))


def _get_default_stack_priors_path() -> str:
    return os.environ.get(
        "ORCHESTRATOR_PATHS_STACK_PRIORS_PATH",
        f"{_get_default_project_root()}/orchestration/derived/stack_priors.yaml",
    )


# 2026-07-30 HALF FLEET. Quarters retired; these last-resort literals dropped
# the freed ports 8280/8380/8282/8382/8385/8485 (and vision_escalation's
# 8187/8287/8387/8487, which were never launched at all — that role is and
# was single-instance). A stale port here is not cosmetic: dispatch fails
# OPEN on a port it cannot resolve to a topology index, so an unknown
# endpoint yields NO region lock rather than an error.
# GATED SURFACE (SSU-F8, 2026-09-23). This table is the SIXTH surface that restates
# `server_mode.<host>.shared_with`, and the only one that used to be unwired -- which is
# why it is the one that rotted. `scripts/validate/check_shared_with_derivations.py` now
# RECOMPUTES each row from the master registry (`server_mode.<host>.port` + `numa_ports`)
# and diffs it. Do not hand-edit a row to today's ports: move the role in the registry,
# run the checker, and copy the value it prints as `expected`.
_LEGACY_SERVER_URL_FALLBACKS: dict[str, str] = {
    "frontdoor": (
        "full:http://localhost:8070,http://localhost:8080,"
        "http://localhost:8180"
    ),
    # 2026-08-01 W1 CUTOVER: coder_escalation left the frontdoor fleet for
    # architect_general's single GPU process (since the 2026-09-27 ARCHITECT SWAP
    # that process serves architect_critic; same :8083). It has no half-fleet siblings — the
    # 27B is one MI210 server, not a 1-full-plus-2-halves CPU lineup, so the
    # "full:" multi-URL form would advertise ports that do not exist.
    "coder_escalation": "http://localhost:8083",
    # 2026-09-22 CUTOVER (row corrected 2026-09-23, SSU-F8): the worker lane
    # -- worker_general, worker_math, toolrunner -- left its own :8072/:8082/:8182
    # server and became ALIASES on frontdoor's :8070 process
    # (server_mode.frontdoor.shared_with). The five wired surfaces were updated at the
    # cutover; these three rows were not, and kept resolving to a retired fleet in
    # exactly the degraded mode this table exists to serve. Value = frontdoor's, by
    # derivation and not by copy.
    "worker_general": (
        "full:http://localhost:8070,http://localhost:8080,"
        "http://localhost:8180"
    ),
    "worker_math": (
        "full:http://localhost:8070,http://localhost:8080,"
        "http://localhost:8180"
    ),
    # 2026-08-03: RESTORED. `ServerURLsConfig.toolrunner` used to ask for the
    # literal "worker_general"; the 2026-08-01 W1 cutover repointed it at its OWN
    # name so the alias resolves through the registry-derived tables — but did not
    # add the matching last-resort literal here. `_server_url_default` ends in a
    # bare subscript of this dict, and `toolrunner` is the one field whose
    # canonical name is itself and has no `_STACK_PRIOR_SERVER_URL_ALIASES` entry,
    # so `ServerURLsConfig()` raised `KeyError: 'toolrunner'` whenever the derived
    # stack priors were missing or unreadable — i.e. exactly the degraded mode this
    # table exists to serve (fresh checkout, bootstrap before the launcher writes
    # runtime facts). That is the hazard the `architect_critic` comment two fields
    # below already warns about. Value mirrors worker_general, its shared process,
    # which is byte-identical to the pre-cutover delegation.
    "toolrunner": (
        "full:http://localhost:8070,http://localhost:8080,"
        "http://localhost:8180"
    ),
    "worker_vision": "http://localhost:8086",
    "vision_escalation": "http://localhost:8086",  # 2026-08-01 W1: alias, same process (was :8087)
    "worker_fast": "http://localhost:8102",
    "worker_summarize": "full:http://localhost:8070,http://localhost:8080,http://localhost:8180",  # frontdoor-fleet alias, parity-guarded
    # 2026-09-27 ARCHITECT SWAP (operator-decided): the role labels swapped over
    # unchanged processes. architect_general = Qwen3.8-Flash-Next UD-IQ4_XS on the
    # full CPU instance (:8074); architect_critic = Qwen3.8-27B Q8 on the MI210
    # (:8083) and host of coder_escalation + ingest_long_context.
    # (architect_critic was added 2026-08-01 (W1) on :8074 as the 122B on CPU.)
    "architect_general": "http://localhost:8074",
    "architect_critic": "http://localhost:8083",
    # 2026-09-23 (SSU-F8): ingest_long_context is an alias on its host's single GPU
    # process (server_mode.<host>.shared_with; host = architect_critic since the
    # 2026-09-27 swap, architect_general before), not a three-
    # instance CPU fleet of its own. The :8085/:8185/:8285 fleet this row named was
    # retired with the role's own server; recomputed from its host it is the bare
    # :8083 the host declares. Found by the sixth-surface check, not by a reader.
    "ingest_long_context": "http://localhost:8083",
    "api_url": "http://localhost:8000",
    "ocr_server": "http://localhost:9001",
    "vision_api": "http://localhost:8000/v1/vision/analyze",
}

_WORKER_CODER_SERVER_URL_ALIAS: dict[str, str] = {
    "worker_coder": "worker_fast",
}
_STACK_PRIOR_SERVER_URL_ALIASES: dict[str, str] = dict(_WORKER_CODER_SERVER_URL_ALIAS)
_STACK_MANIFEST_SERVER_URL_ALIASES: dict[str, str] = dict(_WORKER_CODER_SERVER_URL_ALIAS)
_STACK_MANIFEST_SERVICE_ROLES: dict[str, str] = {
    "api_url": "orchestrator",
    "ocr_server": "document_formalizer",
}
_STACK_PRIOR_SERVER_URLS_CACHE: dict[str, str] | None = None
_CANONICAL_SERVER_URL_ALIASES: dict[str, str] = {
    "coder": "coder_escalation",
    "worker": "worker_general",
}
# The runtime-facts writer records primary live roles.  Alias rows can retain a
# stale PID after an API-only restart, so complete these stable same-process
# aliases before static priors get a chance to reintroduce dead full ports.
_RUNTIME_SELECTED_ROLE_ALIASES: dict[str, str] = {
    # 2026-08-01 W1 CUTOVER: coder_escalation's host moved frontdoor -> architect_general.
    # This table OVERRIDES the registry at runtime, so leaving it stale would land
    # coder_escalation requests back on :8070 no matter what the registry declares.
    # 2026-09-27 ARCHITECT SWAP: the :8083 27B process now serves architect_critic,
    # so the host label is architect_critic (leaving it on architect_general would
    # land coder_escalation on Flash-Next :8074).
    # Kept in lockstep with the identical tables in src/api/routes/health.py and
    # scripts/server/stack_env.py.
    "coder_escalation": "architect_critic",
    "vision_escalation": "worker_vision",
    "worker_summarize": "frontdoor",
    "worker_explore": "worker_general",
    "worker_math": "worker_general",
    "toolrunner": "worker_general",
}
_BOOTSTRAP_QUARTERABLE_HOST_ROLES = frozenset(
    {"frontdoor", "worker_general", "ingest_long_context"}
)
_BOOTSTRAP_RUNTIME_FACTS_SCHEMA_VERSION = 1
_BOOTSTRAP_PORT_PROBE_TIMEOUT_S = 0.15


def _localhost_url_from_port(port: Any) -> str | None:
    return f"http://localhost:{port}" if isinstance(port, int) and port > 0 else None


# ============================================================================
# ESC-8 Fix 5: producer-lineup liveness validation
# ----------------------------------------------------------------------------
# A producer-derived server lineup (env-filter branch OR runtime-facts branch)
# must be VALIDATED against the live fleet before it is trusted. The failure this
# guards: env=full (or a structurally-valid full-mode manifest) resolves every
# hot role to the dead full ports 8070/8072/8085 on a sub-full-only fleet (the
# 'quarter' token = half instances since the 2026-07-30 quarter retirement).
# The probe is a bare TCP connect (localhost, short timeout, cached per-process),
# never an HTTP request to a llama-server. It is injectable/mockable so config
# init stays fast and deterministic in tests.
#
# A total cold start is NOT this failure: with no live fleet anywhere, every
# lineup names not-yet-listening ports, so the veto only fires when at least one
# probed port IS listening (i.e. a live fleet exists that the lineup contradicts).
# ============================================================================

_PORT_LISTENING_CACHE: dict[int, bool] = {}
_PORT_PROBE_HOST = "127.0.0.1"
_PORT_PROBE_TIMEOUT_S = 0.15


def _port_listening(port: int) -> bool:
    """Return True when localhost:port accepts a bare TCP connection (cached).

    Delegates the actual bare-TCP-connect to ``realized_fleet.probe_listening``
    — the single ESC-8-sanctioned socket seam (scripts/server/realized_fleet.py)
    — imported lazily to avoid the scripts.server import cycle documented in the
    audit (src.api -> stack_paths.get_config -> ServerURLsConfig -> stack_manifest
    still initializing). Results are cached per-process;
    ``reset_stack_prior_server_url_cache`` clears the cache so a later resolution
    re-probes the live fleet. This module-level function is the mockable seam:
    tests patch it (or pass ``probe=`` to ``_selected_servers_are_live``) and
    never open a real socket.
    """
    if not isinstance(port, int) or isinstance(port, bool) or port <= 0:
        return False
    cached = _PORT_LISTENING_CACHE.get(port)
    if cached is not None:
        return cached
    result = False
    try:
        from scripts.server.realized_fleet import probe_listening

        result = port in probe_listening(
            [port], host=_PORT_PROBE_HOST, timeout=_PORT_PROBE_TIMEOUT_S
        )
    except Exception:
        result = False
    _PORT_LISTENING_CACHE[port] = result
    return result


def _quarterable_host_roles() -> set[str]:
    """Roles whose serving ports flip between the (dead-in-quarter) full port and
    their quarter siblings — the discriminator for a poisoned lineup."""
    try:
        from scripts.server.stack_numa import NUMA_CONFIG
    except Exception:
        return set()
    if not isinstance(NUMA_CONFIG, dict):
        return set()
    roles: set[str] = set()
    for role, cfg in NUMA_CONFIG.items():
        if (
            isinstance(cfg, dict)
            and "full_instance_idx" in cfg
            and len(cfg.get("instances") or []) > 1
        ):
            roles.add(str(role))
    return roles


def _selected_servers_are_live(
    selected_servers: list[dict[str, Any]] | None,
    *,
    probe: Callable[[int], bool] | None = None,
) -> bool:
    """Return True when a producer lineup is consistent with the live fleet.

    Rejects a lineup in which a quarterable host role (frontdoor /
    worker_general / ingest_long_context) names ONLY dead ports — the exact
    poison signature of an env=full or valid-full-manifest lineup on a
    sub-full-only fleet. A total cold start (nothing listening ANYWHERE — the
    fleet is still booting) is NOT a poison signature: every lineup names
    not-yet-listening ports then, so the lineup is accepted rather than
    vetoed. When the host-role discriminator is unavailable, falls back to
    requiring at least one live port anywhere, so a lineup that names only
    dead ports is never trusted once any port at all is listening.
    """
    if not selected_servers:
        return False
    is_live = probe or _port_listening
    ports_by_role: dict[str, set[int]] = {}
    all_ports: set[int] = set()
    for server in selected_servers:
        if not isinstance(server, dict):
            continue
        port = server.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or port <= 0:
            continue
        all_ports.add(port)
        roles = server.get("roles")
        if isinstance(roles, list):
            for role in roles:
                if isinstance(role, str) and role:
                    ports_by_role.setdefault(role, set()).add(port)
    if not all_ports:
        return False
    # Total cold start: nothing is listening anywhere (the fleet is booting).
    # A lineup naming only not-yet-listening ports is then EXPECTED, not a
    # poison signature — vetoing it would reject the launcher's own lineup and
    # fall through to stale stack priors.
    if not any(is_live(port) for port in sorted(all_ports)):
        return True
    checked_host = False
    for role in _quarterable_host_roles():
        role_ports = ports_by_role.get(role)
        if not role_ports:
            continue
        checked_host = True
        if not any(is_live(port) for port in sorted(role_ports)):
            return False
    if checked_host:
        return True
    return any(is_live(port) for port in sorted(all_ports))


def _bootstrap_port_listening(port: int) -> bool:
    """Bare localhost probe kept independent of the scripts.server import graph."""
    try:
        with socket.create_connection(("127.0.0.1", port), _BOOTSTRAP_PORT_PROBE_TIMEOUT_S):
            return True
    except OSError:
        return False


def _bootstrap_runtime_selected_servers() -> list[dict[str, Any]] | None:
    """Read fresh realized facts without importing ``scripts.server``.

    During API module bootstrap, importing the canonical runtime-facts reader
    can recurse through ``stack_paths -> src.config``.  This small stdlib-only
    reader accepts full, quarter, and both manifests, validates their declared
    mode against launch intent, and probes every quarterable host-role group.
    """
    declared_mode = os.environ.get("ORCHESTRATOR_STACK_NUMA_MODE", "").strip().lower()
    if declared_mode and declared_mode not in {"full", "quarter", "both"}:
        return None
    llm_root = os.environ.get("ORCHESTRATOR_PATHS_LLM_ROOT", "/mnt/raid0/llm")
    tmp_dir = Path(os.environ.get("TMPDIR", f"{llm_root}/tmp"))
    path = tmp_dir / "orchestrator_runtime_facts.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtime_stack = payload.get("runtime_stack")
        if (
            payload.get("schema") != "epyc.orchestrator.runtime_facts"
            or payload.get("schema_version") != _BOOTSTRAP_RUNTIME_FACTS_SCHEMA_VERSION
            or not isinstance(runtime_stack, dict)
        ):
            return None
        manifest_mode = runtime_stack.get("stack_numa_mode")
        if manifest_mode not in {"full", "quarter", "both"}:
            return None
        if declared_mode and manifest_mode != declared_mode:
            return None
        selected_ports = runtime_stack.get("selected_ports")
        selected_servers = runtime_stack.get("selected_servers")
        if not isinstance(selected_ports, list) or not isinstance(selected_servers, list):
            return None
        if any(isinstance(port, bool) or not isinstance(port, int) or port <= 0 for port in selected_ports):
            return None
        declared_ports = set(selected_ports)
        if len(declared_ports) != len(selected_ports):
            return None
        normalized: list[dict[str, Any]] = []
        observed_ports: set[int] = set()
        host_ports: dict[str, set[int]] = {
            role: set() for role in _BOOTSTRAP_QUARTERABLE_HOST_ROLES
        }
        for server in selected_servers:
            if not isinstance(server, dict):
                return None
            port = server.get("port")
            roles = server.get("roles")
            if (
                isinstance(port, bool)
                or not isinstance(port, int)
                or port <= 0
                or port in observed_ports
                or not isinstance(roles, list)
                or not roles
                or any(not isinstance(role, str) or not role for role in roles)
            ):
                return None
            observed_ports.add(port)
            for role in roles:
                canonical_role = _RUNTIME_SELECTED_ROLE_ALIASES.get(role, role)
                if canonical_role in host_ports:
                    host_ports[canonical_role].add(port)
            normalized.append(dict(server))
        if observed_ports != declared_ports:
            return None
        log_dir = Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_LOG_DIR",
                f"{_get_default_project_root()}/logs",
            )
        )
        state_file = log_dir / "orchestrator_state.json"
        if state_file.exists() and path.stat().st_mtime < state_file.stat().st_mtime:
            return None
        for ports in host_ports.values():
            if not ports or not any(_bootstrap_port_listening(port) for port in ports):
                return None
        return normalized
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _runtime_or_env_selected_servers() -> list[dict[str, Any]] | None:
    """Return launcher-selected servers for the active NUMA mode.

    The generated stack-priors file is static and can describe the full-mode
    primary ports even when the live launcher selected quarter mode. Runtime
    facts are authoritative after stack startup; during API startup, before that
    manifest is refreshed, fall back to ORCHESTRATOR_STACK_NUMA_MODE plus the
    stack manifest filter.

    ESC-8 Fix 5: every producer lineup is validated against the live fleet
    before it is returned; a lineup that names only dead ports (env=full on a
    quarters-only fleet, or a poisoned valid full manifest) is rejected and we
    fall through to the next producer (ultimately stack priors).
    """
    # 2026-08-04: this flag suppresses the RUNTIME-FACTS producers only — the
    # bootstrap reader and producer 1, both of which read
    # orchestrator_runtime_facts.json off the live host. It must NOT suppress
    # producer 2 (the env-declared NUMA mode -> static stack-manifest filter),
    # which is a different producer reading a different artifact; the flag name,
    # its docstring, and the producer-2 comment below all draw that line.
    # e923a40b reordered the producers so realized facts outrank launch intent
    # (correct, ESC-8 Fix 5) but hoisted this guard from just above the
    # runtime-facts read to the top of the function, silently widening it to
    # kill every producer. That made the quarter/both-mode tests in
    # tests/unit/test_config.py::TestServerURLsConfig dead by construction —
    # their class fixture sets this flag, so they landed on
    # _LEGACY_SERVER_URL_FALLBACKS and never exercised producer 2 at all.
    # tests/unit/test_full_slot_demotion.py::_fallback_env separately
    # `delenv`s ORCHESTRATOR_STACK_NUMA_MODE to reach the literal fallbacks,
    # which is only meaningful under the narrow scope restored here.
    ignore_runtime_facts = os.environ.get("ORCHESTRATOR_IGNORE_RUNTIME_STACK_FACTS") == "1"

    if not ignore_runtime_facts:
        # Bootstrap producer: an API starts while scripts.server modules can
        # still be partially initialized.  Prefer the fresh launcher artifact over
        # importing that cycle; full-mode behavior remains on the normal chain.
        bootstrap_candidate = _bootstrap_runtime_selected_servers()
        if bootstrap_candidate is not None:
            return bootstrap_candidate

        # Producer 1: validated runtime-facts manifest.  Launcher facts describe
        # realized ports; an inherited env value is launch intent only.
        try:
            from scripts.server.runtime_facts_manifest import (
                read_runtime_stack_selected_servers,
            )

            candidate = read_runtime_stack_selected_servers()
        except Exception as exc:
            _LOGGER.warning(
                "runtime-facts selected-servers read failed (%s: %s)",
                type(exc).__name__,
                exc,
            )
            candidate = None
        if candidate is not None:
            if _selected_servers_are_live(candidate):
                return candidate
            _LOGGER.warning(
                "runtime-facts lineup rejected: host-role ports not listening; "
                "falling through to env/stack priors"
            )

    # Producer 2: env-declared mode → static stack-manifest filter.  This is a
    # startup fallback only; it must never outrank validated realized facts.
    mode = os.environ.get("ORCHESTRATOR_STACK_NUMA_MODE")
    if mode:
        candidate: list[dict[str, Any]] | None = None
        try:
            from scripts.server.stack_manifest import (
                HOT_SERVERS,
                WARM_SERVERS,
                _filter_by_numa_mode,
            )
            from scripts.server.stack_numa_mode import normalize_stack_numa_mode

            candidate = _filter_by_numa_mode(
                HOT_SERVERS + WARM_SERVERS,
                normalize_stack_numa_mode(mode),
            )
        except Exception as exc:
            _LOGGER.warning(
                "ORCHESTRATOR_STACK_NUMA_MODE=%s env-filter branch failed "
                "(%s: %s); falling through to stack-priors",
                mode,
                type(exc).__name__,
                exc,
            )
            candidate = None
        if candidate is not None:
            if _selected_servers_are_live(candidate):
                return candidate
            _LOGGER.warning(
                "ORCHESTRATOR_STACK_NUMA_MODE=%s lineup rejected: host-role ports "
                "not listening (dead full ports while only sub-full ports are "
                "live?); falling through to stack priors",
                mode,
            )
    return None


def _selected_server_url_values(selected_servers: list[dict[str, Any]] | None) -> dict[str, str]:
    """Build config-compatible role URLs from selected launcher servers."""
    if not selected_servers:
        return {}
    try:
        from scripts.server.stack_numa import NUMA_CONFIG
    except Exception:
        NUMA_CONFIG = {}

    ports_by_role: dict[str, list[tuple[int, bool]]] = {}
    seen_by_role: dict[str, set[int]] = {}
    for server in selected_servers:
        if not isinstance(server, dict):
            continue
        port = server.get("port")
        roles = server.get("roles")
        if isinstance(port, bool) or not isinstance(port, int) or port <= 0:
            continue
        if not isinstance(roles, list):
            continue
        for role in roles:
            if not isinstance(role, str) or not role:
                continue
            canonical_role = _RUNTIME_SELECTED_ROLE_ALIASES.get(role, role)
            seen = seen_by_role.setdefault(canonical_role, set())
            if port in seen:
                continue
            seen.add(port)
            # NUMA_CONFIG is keyed by the role that OWNS the topology -- an alias
            # launches no server and therefore has no entry, which is the invariant
            # `check_shared_with_derivations` enforces on stack_topology.yaml. Looking
            # the alias up directly always missed, so `is_full` was False for every
            # port of every alias and the `full:` marker (which arms
            # ConcurrencyAwareBackend's full slot) could never be emitted for one.
            # Harmless while the worker lane hosted its own fleet; a silent demotion
            # since 2026-09-22, when the lane became aliases. The runtime-facts writer
            # already records the owning role per server row as `topology_role`
            # (scripts/server/runtime_facts_manifest.py); this reads it instead of
            # re-deriving it. Falls back to the role's own name when absent, so a row
            # without the key behaves exactly as before.
            topology_role = server.get("topology_role")
            lookup_role = (
                topology_role if isinstance(topology_role, str) and topology_role else canonical_role
            )
            cfg = NUMA_CONFIG.get(lookup_role) if isinstance(NUMA_CONFIG, dict) else None
            full_idx = cfg.get("full_instance_idx") if isinstance(cfg, dict) else None
            is_full = (
                isinstance(full_idx, int)
                and isinstance(server.get("numa_instance"), int)
                and server.get("numa_instance") == full_idx
            )
            ports_by_role.setdefault(canonical_role, []).append((port, is_full))

    urls: dict[str, str] = {}
    for role, entries in ports_by_role.items():
        full_ports = [port for port, is_full in entries if is_full]
        other_ports = [port for port, is_full in entries if not is_full]
        ordered = full_ports[:1] + other_ports + full_ports[1:]
        if not ordered:
            continue
        role_urls = [f"http://localhost:{port}" for port in ordered]
        if full_ports and len(role_urls) > 1:
            role_urls[0] = f"full:{role_urls[0]}"
        urls[role] = ",".join(role_urls)

    for alias, primary_role in _RUNTIME_SELECTED_ROLE_ALIASES.items():
        if alias not in urls and primary_role in urls:
            urls[alias] = urls[primary_role]
    return urls


def _stack_manifest_server_urls() -> dict[str, str]:
    """Return compatibility/service URLs derived from stack manifest ports."""
    try:
        from scripts.server.stack_manifest import PORT_MAP
    except Exception:
        return {}

    urls: dict[str, str] = {}
    worker_fast = _localhost_url_from_port(PORT_MAP.get("worker_fast"))
    if worker_fast:
        urls["worker_fast"] = worker_fast

    for alias, target in _STACK_MANIFEST_SERVER_URL_ALIASES.items():
        if target in urls:
            urls[alias] = urls[target]

    for name, role in _STACK_MANIFEST_SERVICE_ROLES.items():
        url = _localhost_url_from_port(PORT_MAP.get(role))
        if url:
            urls[name] = url

    api_url = urls.get("api_url")
    if api_url:
        urls["vision_api"] = f"{api_url}/v1/vision/analyze"
    return urls


def _stack_prior_server_urls() -> dict[str, str]:
    """Return role URLs derived from generated stack priors, if available."""
    global _STACK_PRIOR_SERVER_URLS_CACHE
    if _STACK_PRIOR_SERVER_URLS_CACHE is not None:
        return _STACK_PRIOR_SERVER_URLS_CACHE

    urls: dict[str, str] = {}
    urls.update(_selected_server_url_values(_runtime_or_env_selected_servers()))
    try:
        priors_path = Path(_get_default_stack_priors_path())
        for role, url in live_stack_serving_url_values(priors_path).items():
            urls.setdefault(role, url)

        for alias, target in _STACK_PRIOR_SERVER_URL_ALIASES.items():
            if target in urls:
                urls[alias] = urls[target]
    except Exception:
        urls = {}

    for name, url in _stack_manifest_server_urls().items():
        urls.setdefault(name, url)

    _STACK_PRIOR_SERVER_URLS_CACHE = urls
    return _STACK_PRIOR_SERVER_URLS_CACHE


def _canonical_server_url_name(name: str) -> str:
    if name in {"worker_coder", "worker_fast"}:
        return name
    return _CANONICAL_SERVER_URL_ALIASES.get(name, str(Role.from_string(name) or name))


def _server_url_default(name: str) -> str:
    urls = _stack_prior_server_urls()
    if name in {"worker_coder", "worker_fast"} and name in urls:
        return urls[name]
    canonical = _canonical_server_url_name(name)
    if canonical in urls:
        return urls[canonical]
    alias = _STACK_PRIOR_SERVER_URL_ALIASES.get(name) or _STACK_MANIFEST_SERVER_URL_ALIASES.get(name)
    if alias and alias in urls:
        return urls[alias]
    return _LEGACY_SERVER_URL_FALLBACKS[alias or canonical]


def reset_stack_prior_server_url_cache() -> None:
    """Reset generated stack-prior server URL defaults."""
    global _STACK_PRIOR_SERVER_URLS_CACHE
    _STACK_PRIOR_SERVER_URLS_CACHE = None
    # ESC-8 Fix 5(d): the per-process port-liveness probe cache is tied to the
    # same resolution; clear it so a re-resolution re-probes the live fleet.
    _PORT_LISTENING_CACHE.clear()


@dataclass
class PathsConfig:
    """Configuration for file paths.

    All paths can be overridden via ORCHESTRATOR_PATHS_* environment variables.
    Default values assume /mnt/raid0/llm layout but can be reconfigured.
    """

    # Base paths (configure these to relocate everything)
    llm_root: Path = field(default_factory=lambda: Path(_get_default_llm_root()))
    """Root directory for all LLM-related files."""

    project_root: Path = field(default_factory=lambda: Path(_get_default_project_root()))
    """Project root directory (claude repo)."""

    # Derived paths - these use llm_root/project_root as base
    models_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ORCHESTRATOR_PATHS_MODELS_DIR", f"{_get_default_llm_root()}/models")
        )
    )
    """Directory for GGUF models."""

    cache_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ORCHESTRATOR_PATHS_CACHE_DIR", f"{_get_default_llm_root()}/cache")
        )
    )
    """Cache directory."""

    tmp_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ORCHESTRATOR_PATHS_TMP_DIR", f"{_get_default_llm_root()}/tmp")
        )
    )
    """Temporary files directory."""

    registry_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_REGISTRY_PATH",
                f"{_get_default_project_root()}/orchestration/model_registry.yaml",
            )
        )
    )
    """Path to model registry YAML."""

    tool_registry_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_TOOL_REGISTRY_PATH",
                f"{_get_default_project_root()}/orchestration/tool_registry.yaml",
            )
        )
    )
    """Path to tool registry YAML."""

    script_registry_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_SCRIPT_REGISTRY_DIR",
                f"{_get_default_project_root()}/orchestration/script_registry",
            )
        )
    )
    """Directory for script registry."""

    sessions_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_SESSIONS_DIR",
                f"{_get_default_project_root()}/orchestration/repl_memory/sessions",
            )
        )
    )
    """Session storage directory."""

    artifacts_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_ARTIFACTS_DIR", f"{_get_default_llm_root()}/tmp/claude/artifacts"
            )
        )
    )
    """Artifacts directory for context manager."""

    llama_cpp_bin: Path = field(
        default_factory=lambda: Path(
            os.environ["ORCHESTRATOR_PATHS_LLAMA_CPP_BIN"]
            if "ORCHESTRATOR_PATHS_LLAMA_CPP_BIN" in os.environ
            else str(_kernel_backend_dir("cpu"))
        )
    )
    """llama.cpp binary directory."""

    model_base: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_MODEL_BASE", f"{_get_default_llm_root()}/models"
            )
        )
    )
    """Base directory for models (consolidated root 2026-07-30; the old
    lmstudio/models root remains a symlink farm)."""

    log_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ORCHESTRATOR_PATHS_LOG_DIR", f"{_get_default_project_root()}/logs")
        )
    )
    """Log files directory."""

    raid_prefix: str = field(
        default_factory=lambda: os.environ.get("ORCHESTRATOR_PATHS_RAID_PREFIX", "/mnt/raid0/")
    )
    """Required prefix for all data paths (security). Set to empty string to disable check."""


@dataclass
class ServerURLsConfig:
    """Server URL mapping for all orchestrator roles.

    Each field maps an orchestrator role to a llama-server URL.
    Generated stack priors are the primary source of truth; literal values
    remain as degraded fallbacks when the generated artifact is unavailable.
    """

    # Tier A - Front Door. "full:" prefix triggers ConcurrencyAwareBackend.
    frontdoor: str = field(default_factory=lambda: _server_url_default("frontdoor"))

    # Tier B - Specialists.
    #
    # 2026-08-01 W1 CUTOVER — `coder_escalation` NO LONGER DELEGATES TO FRONTDOOR.
    # The comment that stood here said "server_mode.coder_escalation is pinned to
    # frontdoor port 8070, shared mmap". That ceased to be true at the cutover: the
    # role moved to architect_general's MI210 :8083 process and is a different model
    # on a different device. (2026-09-27 ARCHITECT SWAP: that same :8083 process now
    # serves architect_critic; the field still asks for its OWN name, so the
    # resolver follows the host through `shared_with` with no edit here.)
    # This was the LAST copy still pointing at the old host. `PORT_MAP`,
    # `_LEGACY_SERVER_URL_FALLBACKS` and `_RUNTIME_SELECTED_ROLE_ALIASES` were all
    # updated in the cutover commit; this field default was missed, and because it
    # asks for the literal string "frontdoor" it BYPASSED the correct resolution —
    # `_stack_prior_server_urls()["coder_escalation"]` already returned :8083 while
    # `ServerURLsConfig().coder_escalation` still returned frontdoor's `full:` fleet.
    # Since `as_dict()` here is the backend map for chat, routing, openai-compat and
    # health, every coder_escalation request was still landing on frontdoor's CPU
    # 35B — i.e. the null hop the whole cutover existed to remove.
    coder_escalation: str = field(default_factory=lambda: _server_url_default("coder_escalation"))
    # `coder` is a CANDIDATE-ROLE label, not a serving role — it has no server_mode
    # row and no port of its own, so delegating its URL default is correct. It stays
    # on frontdoor: D3 removed `coder` from frontdoor's candidate_roles for AUTOPILOT
    # ARM PURPOSES, but frontdoor is still hop 1 of escalation_chains.coder and is
    # still the cheap CPU lane a bare `coder` label should resolve to.
    coder: str = field(default_factory=lambda: _server_url_default("frontdoor"))

    # Tier C - Workers. `worker` and deprecated worker_* roles stay as
    # compatibility aliases where stack priors do not expose that exact label.
    worker: str = field(default_factory=lambda: _server_url_default("worker"))
    worker_general: str = field(default_factory=lambda: _server_url_default("worker_general"))
    # 2026-08-01: these ask for their OWN role name, not their alias host's.
    # `_server_url_default` already resolves an alias to its serving process via
    # `_STACK_PRIOR_SERVER_URL_ALIASES` / `_STACK_MANIFEST_SERVER_URL_ALIASES`, both
    # of which are derived from the registry's `shared_with` relation — so naming the
    # host here added nothing and broke the moment a role was repointed. That is
    # exactly how `coder_escalation` kept resolving to frontdoor's :8070 through the
    # W1 cutover: the VALUE was derived, but the LOOKUP KEY was hardcoded, so the
    # resolver faithfully returned the right answer to the wrong question.
    # Verified byte-identical to the previous hardcoded delegation before changing.
    toolrunner: str = field(default_factory=lambda: _server_url_default("toolrunner"))
    worker_explore: str = field(default_factory=lambda: _server_url_default("worker_explore"))
    worker_math: str = field(default_factory=lambda: _server_url_default("worker_math"))
    worker_vision: str = field(default_factory=lambda: _server_url_default("worker_vision"))
    vision_escalation: str = field(
        default_factory=lambda: _server_url_default("vision_escalation")
    )
    worker_coder: str = field(default_factory=lambda: _server_url_default("worker_coder"))
    worker_fast: str = field(default_factory=lambda: _server_url_default("worker_fast"))
    # 2026-08-01: asks for its own name; the resolver derives the frontdoor host from
    # `server_mode.frontdoor.shared_with`. Verified byte-identical to the previous
    # hardcoded `_server_url_default("frontdoor")` before changing.
    worker_summarize: str = field(default_factory=lambda: _server_url_default("worker_summarize"))

    # Tier B - Architect / long-context.
    architect_general: str = field(default_factory=lambda: _server_url_default("architect_general"))
    # NEW 2026-08-01 (W1). _server_url_default() ends in a bare subscript of
    # _LEGACY_SERVER_URL_FALLBACKS (models.py:754, no .get/default), so adding a
    # field here WITHOUT the matching fallback entry raises KeyError at config
    # CONSTRUCTION — fresh checkout, degraded mode, and every test that builds a
    # config. The entry was added above in the same commit.
    architect_critic: str = field(default_factory=lambda: _server_url_default("architect_critic"))
    ingest_long_context: str = field(
        default_factory=lambda: _server_url_default("ingest_long_context")
    )

    # Services
    api_url: str = field(default_factory=lambda: _server_url_default("api_url"))
    ocr_server: str = field(default_factory=lambda: _server_url_default("ocr_server"))
    vision_api: str = field(default_factory=lambda: _server_url_default("vision_api"))

    def as_dict(self) -> dict[str, str]:
        """Return role->URL mapping as dict (for LLMPrimitives compatibility).

        Excludes service URLs (api_url, ocr_server, vision_api).
        """
        d = asdict(self)
        # Remove non-role entries
        for key in ("api_url", "ocr_server", "vision_api"):
            d.pop(key, None)
        return d


@dataclass
class TimeoutsConfig:
    """All timeout values in seconds.

    Source of truth: orchestration/model_registry.yaml (runtime_defaults.timeouts)
    Hardcoded defaults here are fallbacks only - registry values take precedence.
    """

    # Role-specific request timeouts (read from registry.timeouts.roles.*)
    worker_explore: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_general", 60))
    )
    worker_math: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_math", 60))
    )
    worker_vision: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_vision", 60))
    )
    worker_summarize: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_summarize", 120))
    )
    worker_general: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_general", 60))
    )
    worker_coder: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_fast", 30))
    )
    worker_code: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_coder", 30))
    )
    worker_fast: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "worker_fast", 30))
    )
    frontdoor: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "frontdoor", 90))
    )
    coder_escalation: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "coder_escalation", 120))
    )
    vision_escalation: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "vision_escalation", 120))
    )
    ingest_long_context: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "ingest_long_context", 300))
    )
    architect_general: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "architect_general", 600))
    )
    architect_critic: int = field(
        default_factory=lambda: int(_registry_timeout("roles", "architect_critic", 600))
    )
    default_request: int = field(
        default_factory=lambda: int(_registry_timeout("", "default", 600))
    )
    """Fallback for unknown roles."""

    # Backend timeouts (read from registry.timeouts.server.*)
    server_request: int = field(
        default_factory=lambda: int(_registry_timeout("server", "request", 600))
    )
    server_connect: int = field(
        default_factory=lambda: int(_registry_timeout("server", "connect", 5))
    )

    # Service timeouts (read from registry.timeouts.services.*)
    ocr_single_page: float = field(
        default_factory=lambda: float(_registry_timeout("services", "ocr_single_page", 120.0))
    )
    ocr_pdf: float = field(
        default_factory=lambda: float(_registry_timeout("services", "ocr_pdf", 600.0))
    )
    health_check: float = field(
        default_factory=lambda: float(_registry_timeout("server", "health_check", 5.0))
    )
    vision_inference: int = field(
        default_factory=lambda: int(_registry_timeout("services", "vision_inference", 120))
    )
    vision_figure: float = field(
        default_factory=lambda: float(_registry_timeout("services", "vision_figure", 60.0))
    )
    ffmpeg_version: int = field(
        default_factory=lambda: int(_registry_timeout("services", "ffmpeg_version", 5))
    )
    ffmpeg_probe: int = field(
        default_factory=lambda: int(_registry_timeout("services", "ffmpeg_probe", 30))
    )
    ffmpeg_extract: int = field(
        default_factory=lambda: int(_registry_timeout("services", "ffmpeg_extract", 600))
    )
    exiftool: int = field(
        default_factory=lambda: int(_registry_timeout("services", "exiftool", 30))
    )
    gradio_client: float = field(
        default_factory=lambda: float(_registry_timeout("services", "gradio_client", 300.0))
    )

    def _timeout_role_map(self) -> dict[str, int]:
        """Return the canonical role->timeout mapping used by lookup helpers."""
        return {
            "worker_explore": self.worker_general,
            "worker_math": self.worker_math,
            "worker_vision": self.worker_vision,
            "worker_summarize": self.worker_summarize,
            "worker_general": self.worker_general,
            "worker_coder": self.worker_coder,
            "worker_code": self.worker_code,
            "worker_fast": self.worker_fast,
            "frontdoor": self.frontdoor,
            "coder_escalation": self.coder_escalation,
            "vision_escalation": self.vision_escalation,
            "ingest_long_context": self.ingest_long_context,
            "architect_general": self.architect_general,
            "architect_critic": self.architect_critic,
        }

    def _normalize_timeout_role(self, role: str) -> str:
        """Normalize a timeout lookup role without changing compatibility aliases."""
        if role == "worker_fast":
            return "worker_fast"
        canonical = Role.from_string(role)
        return canonical.value if canonical is not None else role

    def for_role(self, role: str) -> int:
        """Get timeout for a specific role, falling back to default."""
        normalized = self._normalize_timeout_role(role)
        return self._timeout_role_map().get(normalized, self.default_request)

    def role_timeouts_dict(self) -> dict[str, int]:
        """Return role->timeout dict (for backward compat with ROLE_TIMEOUTS)."""
        return self._timeout_role_map()


@dataclass
class VisionConfig:
    """Vision pipeline configuration."""

    base_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ORCHESTRATOR_PATHS_VISION_DIR", f"{_get_default_llm_root()}/vision")
        )
    )
    llama_mtmd_cli: Path = field(
        default_factory=lambda: Path(
            os.environ["ORCHESTRATOR_PATHS_LLAMA_MTMD"]
            if "ORCHESTRATOR_PATHS_LLAMA_MTMD" in os.environ
            else str(_kernel_backend_dir("cpu") / "llama-mtmd-cli")
        )
    )
    # 2026-08-01 W1 CUTOVER. This is a SECOND, OFFLINE vision pipeline
    # (llama-mtmd-cli batch analyzer) with its own model paths and its own port
    # pair, independent of stack priors. It was still pinned to the retired
    # Qwen2.5-VL-7B and to :8087, a port that no longer exists after the vision
    # unification — so the batch analyzer would have kept running the retired
    # model against a dead port while the served stack was correct.
    vl_model_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_VL_MODEL",
                f"{_get_default_llm_root()}/models/lmstudio-community/"
                "Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf",
            )
        )
    )
    vl_mmproj_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_VL_MMPROJ",
                f"{_get_default_llm_root()}/models/lmstudio-community/"
                "Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf",
            )
        )
    )
    vl_server_port: int = 8086
    # Both roles now resolve to the same :8086 process.
    vl_escalation_server_port: int = 8086

    # Processing limits
    max_image_size_mb: int = 20
    max_image_dimension: int = 4096
    default_batch_size: int = 100
    max_concurrent_workers: int = 4
    default_video_fps: float = 1.0
    default_vl_max_tokens: int = 1024  # S-15: floor, reasoning VL models truncate below it
    default_vl_threads: int = 8

    # Thumbnail settings
    thumb_size: tuple[int, int] = (256, 256)
    thumb_quality: int = 85
    temp_jpeg_quality: int = 95

    # Face detection
    face_min_confidence: float = 0.9
    face_embedding_dim: int = 512
    face_identification_threshold: float = 0.6

    # Model names
    arcface_model_name: str = "buffalo_l"
    clip_model_name: str = "ViT-B/32"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    onnx_providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    supported_image_extensions: list[str] = field(
        default_factory=lambda: ["jpg", "jpeg", "png", "heic", "webp", "bmp", "tiff"]
    )


@dataclass
class ChatPipelineConfig:
    """Chat pipeline configuration thresholds."""

    # Three-stage summarization
    summarization_threshold_tokens: int = 5000
    """~20K chars triggers Stage 1+2."""
    multi_doc_discount: float = 0.7
    """Lower threshold for multiple documents."""
    compression_enabled: bool = False
    """Stage 0 compression (disabled due to LLMLingua-2 quality issues)."""
    compression_min_chars: int = 30000
    compression_target_ratio: float = 0.5
    stage1_context_limit: int = 20000

    # Long context exploration
    long_context_enabled: bool = True
    long_context_threshold_chars: int = 20000
    """~5K tokens triggers exploration mode."""
    long_context_max_turns: int = 8

    # Quality detection thresholds
    repetition_unique_ratio: float = 0.5
    garbled_short_line_ratio: float = 0.6
    min_answer_length: int = 50

    # Plan-review Q-value skip threshold (the answer-review gate's
    # review_low_q_threshold was removed by RI-18c, the RI-18 DROP)
    review_skip_q_threshold: float = 0.6

    # Plan review phase transitions
    plan_review_phase_a_min: int = 50
    plan_review_phase_b_mean_q: float = 0.7
    plan_review_phase_b_min_q: float = 0.5
    plan_review_phase_c_min_q: float = 0.7
    plan_review_phase_c_min_total: int = 100
    plan_review_phase_c_skip_rate: float = 0.90
    """Fraction of reviews skipped in Phase C (spot-check)."""

    # Session compaction tuning (C1 virtual memory pattern)
    session_compaction_keep_recent_ratio: float = 0.20
    """Fraction of context to keep verbatim after compaction (default 20%, min 3000 chars)."""
    session_compaction_recompaction_interval: int = 0
    """Re-trigger compaction every N turns after first compaction. 0 = disabled."""
    session_compaction_min_turns: int = 5
    """Minimum turns before compaction can run (default 5 to avoid very-early churn)."""
    session_compaction_trigger_ratio: float = 0.75
    """Fraction of model max context at which compaction fires (default 0.75, was 0.60)."""

    # Try-cheap-first: speculative pre-filter using 7B worker before specialist.
    # Phase A = try all, Phase B = MemRL-guided, Phase C = fully learned.
    try_cheap_first_enabled: bool = True
    try_cheap_first_phase: str = "A"
    """A = try all non-forced, B = Q-value guided, C = fully learned."""
    try_cheap_first_role: str = "worker_general"
    """Role used for cheap attempts (fastest HOT model)."""
    try_cheap_first_max_tokens: int = 1024
    """Token budget for cheap attempt (keep short to minimize waste)."""
    try_cheap_first_quality_threshold: float = 0.6
    """Minimum quality score to accept cheap answer."""
    try_cheap_first_q_threshold: float = 0.65
    """Min Q-value for cheap-first in Phase B/C (skip cheap if below)."""


@dataclass
class DelegationConfig:
    """Configuration for proactive delegation."""

    max_iterations: int = 3
    max_total_iterations: int = 10
    max_concurrent_analysis: int = 4
    max_review_tokens: int = 128
    max_taskir_tokens: int = 256
    max_plan_review_tokens: int = 128
    review_output_max_chars: int = 4000
    """EVL-42 1c-fix (b): char budget for the candidate output shown to the
    reviewer in ``ArchitectReviewService.review()``. Over-budget outputs are
    condensed head+tail with an explicit elision marker (never a silent prefix
    cut), so a long-but-correct answer is not judged on its first 500 chars."""


@dataclass
class MemRLRetrievalConfigData:
    """Configuration for MemRL retrieval/risk/prior tuning knobs."""

    semantic_k: int = 20
    min_similarity: float = 0.3
    min_q_value: float = 0.3
    q_weight: float = 0.7
    cost_lambda: float = 0.15
    cost_tau: float = 1.0
    top_n: int = 5
    confidence_threshold: float = 0.6
    confidence_estimator: str = "median"
    confidence_trim_ratio: float = 0.2
    confidence_min_neighbors: int = 3
    calibrated_confidence_threshold: float | None = None
    conformal_margin: float = 0.0
    risk_control_enabled: bool = False
    risk_budget_id: str = "default"
    risk_gate_min_samples: int = 3
    risk_abstain_target_role: str = "architect_general"
    risk_gate_rollout_ratio: float = 1.0
    risk_gate_kill_switch: bool = False
    risk_budget_guardrail_min_events: int = 50
    risk_budget_guardrail_max_abstain_rate: float = 0.60
    prior_strength: float = 0.15
    warm_probability_hit: float = 0.8
    warm_probability_miss: float = 0.2
    warm_cost_fallback_s: float = 1.0
    cold_cost_fallback_s: float = 3.0


@dataclass
class ThinkHarderConfigData:
    """Configuration for think-harder regulation knobs."""

    min_expected_roi: float = 0.02
    min_samples: int = 5
    cooldown_turns: int = 2
    ema_alpha: float = 0.25
    min_marginal_utility: float = 0.0
    token_budget_min: int = 2048
    token_budget_max: int = 4096
    token_budget_fallback: int = 4096
    temperature_min: float = 0.30
    temperature_max: float = 0.50
    cot_roi_threshold: float = 0.35
    token_penalty_per_4k: float = 0.15
    ema_alpha_min: float = 0.05
    ema_alpha_max: float = 1.0


@dataclass
class ServicesConfig:
    """Configuration for services (OCR, PDF, archives, drafts)."""

    lightonocr_model: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_LIGHTONOCR_MODEL",
                f"{_get_default_llm_root()}/models/LightOnOCR-2-1B-bbox-Q4_K_M.gguf",
            )
        )
    )
    lightonocr_mmproj: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_LIGHTONOCR_MMPROJ",
                f"{_get_default_llm_root()}/models/LightOnOCR-2-1B-bbox-mmproj-F16.gguf",
            )
        )
    )
    lightonocr_max_tokens: int = 2048

    draft_cache_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_DRAFT_CACHE", f"{_get_default_llm_root()}/cache/drafts"
            )
        )
    )
    draft_cache_ttl_hours: float = 24.0

    archive_extract_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_ARCHIVE_EXTRACT",
                f"{_get_default_project_root()}/tmp/archives",
            )
        )
    )
    max_archive_size: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_SERVICES_MAX_ARCHIVE_SIZE", str(500 * 1024 * 1024)))
    )
    max_extracted_size: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_SERVICES_MAX_EXTRACTED_SIZE", str(1024 * 1024 * 1024)))
    )
    max_archive_files: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_SERVICES_MAX_ARCHIVE_FILES", "1000"))
    )
    max_archive_single_file: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_SERVICES_MAX_ARCHIVE_SINGLE_FILE", str(100 * 1024 * 1024)))
    )
    max_archive_compression_ratio: float = field(
        default_factory=lambda: float(os.environ.get("ORCHESTRATOR_SERVICES_MAX_ARCHIVE_COMPRESSION_RATIO", "100.0"))
    )
    max_archive_recursion_depth: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_SERVICES_MAX_ARCHIVE_RECURSION_DEPTH", "2"))
    )

    pdf_min_entropy: float = field(
        default_factory=lambda: float(os.environ.get("ORCHESTRATOR_SERVICES_PDF_MIN_ENTROPY", "3.5"))
    )
    pdf_max_garbage_ratio: float = field(
        default_factory=lambda: float(os.environ.get("ORCHESTRATOR_SERVICES_PDF_MAX_GARBAGE_RATIO", "0.15"))
    )
    pdf_min_word_length_avg: float = field(
        default_factory=lambda: float(os.environ.get("ORCHESTRATOR_SERVICES_PDF_MIN_WORD_LENGTH_AVG", "2.5"))
    )
    pdf_min_text_length: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_SERVICES_PDF_MIN_TEXT_LENGTH", "100"))
    )
    pdftotext_timeout_seconds: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_SERVICES_PDFTOTEXT_TIMEOUT_SECONDS", "30"))
    )

    pdf_router_temp_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_PDF_ROUTER_TEMP", f"{_get_default_llm_root()}/tmp/pdf_router"
            )
        )
    )

    llm_cache_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_LLM_CACHE",
                f"{_get_default_project_root()}/cache/llm_responses",
            )
        )
    )
    """Content-addressable LLM response cache directory."""

    llm_cache_ttl_hours: float = 168.0
    """Cache TTL in hours (default 1 week)."""

    llm_cache_max_entries: int = 10_000
    """Maximum cache entries before LRU eviction."""


@dataclass
class ApiConfig:
    """Configuration for API middleware (CORS, rate limiting)."""

    cors_origins: list[str] = field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ]
    )
    """Allowed CORS origins. Must be explicit when allow_credentials is True."""

    cors_allow_credentials: bool = True
    """Whether to allow credentials in CORS requests."""

    rate_limit_rpm: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_API_RATE_LIMIT_RPM", "60"))
    )
    """Requests per minute per client IP."""

    rate_limit_burst: int = field(
        default_factory=lambda: int(os.environ.get("ORCHESTRATOR_API_RATE_LIMIT_BURST", "10"))
    )
    """Maximum burst size above the sustained rate."""

    rate_limit_cleanup_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("ORCHESTRATOR_API_RATE_LIMIT_CLEANUP_INTERVAL_SECONDS", "300.0")
        )
    )
    """Cleanup cadence for stale rate-limit buckets."""

    rate_limit_stale_bucket_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("ORCHESTRATOR_API_RATE_LIMIT_STALE_BUCKET_TTL_SECONDS", "600.0")
        )
    )
    """Idle TTL before a client bucket is pruned."""


@dataclass
class SessionPersistenceConfigData:
    """Configuration for session checkpoint/summary cadence."""

    checkpoint_turn_interval: int = 5
    checkpoint_idle_minutes: int = 30
    summary_idle_hours: int = 2
    checkpoint_globals_warn_mb: int = 50
    checkpoint_globals_hard_mb: int = 100


@dataclass
class SessionLifecycleConfigData:
    """Configuration for session status transitions."""

    active_to_idle_hours: float = 1.0
    idle_to_stale_days: float = 7.0


@dataclass
class HealthTrackerConfigData:
    """Configuration for backend health tracker circuit breaker."""

    default_failure_threshold: int = 3
    default_cooldown_s: float = 30.0
    max_cooldown_s: float = 300.0


@dataclass
class ExternalAPIConfig:
    """Configuration for a single external API backend."""

    api_key: str = ""
    """API key (loaded from environment)."""

    base_url: str = ""
    """Base URL for the API."""

    default_model: str = ""
    """Default model name to use."""

    timeout: int = 120
    """Request timeout in seconds."""

    max_retries: int = 3
    """Maximum retries on transient failures."""


@dataclass
class ExternalBackendsConfig:
    """Configuration for external API backends (Anthropic, OpenAI, etc.).

    API keys are loaded from environment variables:
      - ANTHROPIC_API_KEY
      - OPENAI_API_KEY

    Usage:
        config = get_config()
        if config.external_backends.anthropic.api_key:
            backend = AnthropicBackend(config.external_backends.anthropic)
    """

    anthropic: ExternalAPIConfig = field(
        default_factory=lambda: ExternalAPIConfig(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            base_url="https://api.anthropic.com",
            default_model="claude-3-5-sonnet-20241022",
            timeout=120,
            max_retries=3,
        )
    )
    """Anthropic API configuration."""

    openai: ExternalAPIConfig = field(
        default_factory=lambda: ExternalAPIConfig(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url="https://api.openai.com/v1",
            default_model="gpt-4o",
            timeout=120,
            max_retries=3,
        )
    )
    """OpenAI API configuration."""

    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic.api_key)

    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai.api_key)


@dataclass
class WorkerPoolPathsConfig:
    """Paths for worker pool management."""

    llama_server_path: Path = field(
        default_factory=lambda: Path(
            os.environ["ORCHESTRATOR_PATHS_LLAMA_SERVER"]
            if "ORCHESTRATOR_PATHS_LLAMA_SERVER" in os.environ
            else str(_kernel_server_binary("cpu"))
        )
    )
    log_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ORCHESTRATOR_PATHS_LOG_DIR", f"{_get_default_project_root()}/logs")
        )
    )
    model_base: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_MODEL_BASE", f"{_get_default_llm_root()}/models"
            )
        )
    )


# NOTE: FeaturesConfig is kept for backward compatibility but features
# should be managed via src.features.Features (its own lifecycle/singleton).
@dataclass
class FeaturesConfig:
    """Configuration for feature flags (DEPRECATED: use src.features instead)."""

    memrl: bool = False
    tools: bool = False
    scripts: bool = False
    streaming: bool = False
    openai_compat: bool = False
    repl: bool = True
    caching: bool = True


@dataclass
class OrchestratorConfigData:
    """Root configuration for the orchestrator system.

    This dataclass provides the complete configuration hierarchy.
    For production use with environment variables, use get_config().
    """

    mock_mode: bool = True
    """Use mock responses instead of real inference."""

    debug: bool = False
    """Enable debug logging."""

    # Existing sections
    llm: LLMConfig = field(default_factory=LLMConfig)
    escalation: EscalationConfigData = field(default_factory=EscalationConfigData)
    repl: REPLConfigData = field(default_factory=REPLConfigData)
    server: ServerConfigData = field(default_factory=ServerConfigData)
    monitor: MonitorConfigData = field(default_factory=MonitorConfigData)
    paths: PathsConfig = field(default_factory=PathsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)

    # New sections (Phase 3)
    server_urls: ServerURLsConfig = field(default_factory=ServerURLsConfig)
    timeouts: TimeoutsConfig = field(default_factory=TimeoutsConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    chat: ChatPipelineConfig = field(default_factory=ChatPipelineConfig)
    delegation: DelegationConfig = field(default_factory=DelegationConfig)
    memrl_retrieval: MemRLRetrievalConfigData = field(default_factory=MemRLRetrievalConfigData)
    think_harder: ThinkHarderConfigData = field(default_factory=ThinkHarderConfigData)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    worker_pool: WorkerPoolPathsConfig = field(default_factory=WorkerPoolPathsConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    session_persistence: SessionPersistenceConfigData = field(
        default_factory=SessionPersistenceConfigData
    )
    session_lifecycle: SessionLifecycleConfigData = field(default_factory=SessionLifecycleConfigData)
    health_tracker: HealthTrackerConfigData = field(default_factory=HealthTrackerConfigData)
    external_backends: ExternalBackendsConfig = field(default_factory=ExternalBackendsConfig)
