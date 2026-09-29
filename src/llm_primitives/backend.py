"""Backend management for LLM primitives."""

import logging
import os
from typing import Any, Mapping

_log = logging.getLogger(__name__)


def _certified_native_batch_width(
    owner: Any,
    *,
    topology_role: str,
    full_url: str | None,
    logical_role: str | None = None,
) -> int:
    """Return the realized shared-lease width for a full CPU endpoint.

    Shared region locks are safe only among aliases of the same physical
    server.  Certification is therefore explicit by topology role and capped
    by both the registry's full-shape slot declaration and live admission.
    """
    allowed = {
        role.strip()
        for role in os.environ.get(
            "ORCHESTRATOR_NATIVE_BATCH_SHARED_ROLES", "frontdoor,worker_general"
        ).split(",")
        if role.strip()
    }
    admitted_role = logical_role or topology_role
    if admitted_role not in allowed or not full_url:
        return 1
    if topology_role == "eval_batch_frontdoor":
        try:
            from scripts.server.stack_manifest import resolve_slots

            declared = resolve_slots(
                "eval_batch_frontdoor", "eval_batch_frontdoor"
            ).slots
        except Exception:
            return 1
        admission = getattr(owner, "admission_controller", None)
        limits = getattr(admission, "_limits", {}) if admission is not None else {}
        try:
            admitted = int(limits.get(full_url, declared))
        except (TypeError, ValueError):
            admitted = declared
        return max(1, min(int(declared), admitted))
    registry = getattr(owner, "registry", None)
    raw = getattr(registry, "_raw", {}) if registry is not None else {}
    server_mode = (raw.get("server_mode") or {}) if isinstance(raw, dict) else {}
    server = server_mode.get(topology_role, {}) if isinstance(server_mode, dict) else {}
    if not server and isinstance(server_mode, dict):
        server = next(
            (
                cfg
                for name, cfg in server_mode.items()
                if isinstance(cfg, dict)
                and (
                    cfg.get("model_role") == topology_role
                    or topology_role in (cfg.get("shared_with") or [])
                    or (name == "worker" and topology_role == "worker_general")
                )
            ),
            {},
        )
    serving_shape = server.get("serving_shape") if isinstance(server, dict) else {}
    slots_by_shape = (
        serving_shape.get("slots_by_shape") if isinstance(serving_shape, dict) else {}
    )
    try:
        declared = int((slots_by_shape or {}).get("full", server.get("slots", 1)))
    except (TypeError, ValueError):
        return 1
    admission = getattr(owner, "admission_controller", None)
    limits = getattr(admission, "_limits", {}) if admission is not None else {}
    try:
        admitted = int(limits.get(full_url, declared))
    except (TypeError, ValueError):
        admitted = declared
    return max(1, min(declared, admitted))


def _url_str_ports(url_str: str) -> list[int]:
    """Ports named by a config URL value (``full:`` marker ignored)."""
    ports: list[int] = []
    for url in _normalise_role_urls(url_str):
        tail = url.rsplit(":", 1)[-1]
        try:
            ports.append(int(tail))
        except ValueError:
            continue
    return ports


def _normalise_role_urls(url_str: str) -> tuple[str, ...]:
    """Return comparable server URLs with the `full:` marker stripped."""
    urls = [u.strip() for u in url_str.split(",") if u.strip()]
    if urls and urls[0].startswith("full:"):
        urls[0] = urls[0][len("full:"):]
    return tuple(urls)


def _infer_topology_role_for_urls(
    role: str,
    role_urls: Mapping[str, tuple[str, ...]],
    topology_roles: set[str],
) -> str:
    """Resolve logical alias roles to the physical topology role they share.

    Some config roles are logical aliases over the same llama-server pool
    (for example coder_escalation shares frontdoor ports). The region-lock
    topology is keyed by the physical pool role, so those aliases must dispatch
    under the canonical topology role while keeping their logical role for
    routing/tap metadata.
    """
    if role in topology_roles:
        return role
    urls = role_urls.get(role, ())
    if not urls:
        return role
    for candidate, candidate_urls in role_urls.items():
        if candidate == role:
            continue
        if candidate in topology_roles and candidate_urls == urls:
            return candidate
    return role


class BackendMixin:
    """Mixin for backend initialization and access methods."""

    def _init_caching_backends(self, server_urls: dict[str, str], num_slots: int) -> None:
        """Initialize CachingBackend instances for each role.

        Args:
            server_urls: Dict mapping role names to llama-server URLs.
                         Values may be comma-separated for multi-instance roles
                         (e.g. "http://localhost:8080,http://localhost:8180").
            num_slots: Number of slots per server.
        """
        try:
            from src.backends.llama_server import LlamaServerBackend, ServerConfig
            from src.backends.round_robin import RoundRobinBackend
            from src.backends.concurrency_aware import ConcurrencyAwareBackend
            from src.prefix_cache import CachingBackend, PrefixRouter

            # 2026-05-23: roles in this env var route through
            # /v1/chat/completions instead of /completion. Used for models
            # whose chat template needs server-side jinja application
            # (gemma-4 multi-channel format, etc.). Comma-separated names.
            # Default: worker_general (gemma-4-26B-A4B-it).
            # J12 (2026-05-26): frontdoor/coder_escalation/architect_general route through
            # /v1/chat/completions so their registry chat_template_kwargs.enable_thinking=false
            # applies (load-bearing, feedback_qwen3x_enable_thinking_false). On /completion the
            # kwarg is inert and Qwen3.6/3.5 emit <think> blocks (J1: degenerate 1390-token
            # thinking on hard prompts). VERIFIED (max_turns=4, reading `answer`): chat-completions
            # frontdoor answers a moderate prompt in 270 tokens / 1 turn, no <think>, vs /completion
            # 2643 tokens / 4 turns thinking-on — ~10x fewer tokens, single turn. (An earlier
            # "0 tokens" reading was a max_turns=1 probe artifact, not a backend bug.)
            # ingest_long_context is EXCLUDED — thinking-on is load-bearing for Qwen3-Next-80B.
            # RI-23: bake only the STATIC set. Flag-admitted thinking roles
            # (thinking_roles_chat_lane) are routed per request inside
            # LlamaServerBackend, so a live flag flip cannot strand a baked backend.
            from src.chat_completions_roles import static_chat_completions_roles
            _chat_completion_roles = static_chat_completions_roles()  # shared SoT (was a divergent inline default)
            if _chat_completion_roles:
                _log.info(
                    "Roles using /v1/chat/completions backend: %s",
                    sorted(_chat_completion_roles),
                )

            # WP-12 fleet layer (ORCHESTRATOR_FLEET_LAYER=1, default OFF):
            # roles bound to a registry server_mode fleet share ONE backend
            # per physical fleet instead of N per-role copies. Roles the
            # fleet path handled are skipped by the legacy loop below; any
            # fleet-layer failure falls back to the legacy build (logged
            # CRITICAL inside). Flag off → empty set, loop untouched.
            fleet_handled: frozenset[str] = frozenset()
            if os.environ.get("ORCHESTRATOR_FLEET_LAYER") == "1":
                fleet_handled = self._init_fleet_backends(
                    server_urls, num_slots, _chat_completion_roles
                )

            normalized_role_urls = {
                role: _normalise_role_urls(url_str)
                for role, url_str in server_urls.items()
            }
            try:
                from src.runtime.instance_topology import get_instance_regions

                topology_roles = {
                    role
                    for role, _idx in get_instance_regions().keys()
                }
            except Exception:
                topology_roles = set()

            # ONE PrefixRouter per PHYSICAL server URL, never per role.
            #
            # Under the registry `shared_with` mechanism several logical roles
            # resolve to the same base_url. A per-role router allocates id_slot in
            # [0, num_slots) from its own private LRU, so two roles on one server
            # both emit id_slot=0 and silently evict each other's KV -- and neither
            # router can observe it, because each sees only its own hit counters.
            #
            # The fleet layer (_init_fleet_backends) already shares one CachingBackend
            # per fleet; this gives the legacy path the same slot-allocation coherence
            # without requiring ORCHESTRATOR_FLEET_LAYER=1.
            _routers_by_url: dict[str, Any] = {}

            def _router_for(endpoint_url: str) -> Any:
                router = _routers_by_url.get(endpoint_url)
                if router is None:
                    router = PrefixRouter(num_slots=num_slots)
                    _routers_by_url[endpoint_url] = router
                return router

            for role, url_str in server_urls.items():
                if role in fleet_handled:
                    continue
                urls = [u.strip() for u in url_str.split(",") if u.strip()]
                topology_role = _infer_topology_role_for_urls(
                    role,
                    normalized_role_urls,
                    topology_roles,
                )

                # Pre-warm convention: first URL prefixed with "full:" denotes
                # the full-speed (1×96t) instance for ConcurrencyAwareBackend.
                # Remaining URLs are quarter (48t) instances.
                has_full = urls and urls[0].startswith("full:")
                if has_full:
                    full_url = urls[0][len("full:"):]
                    quarter_urls = urls[1:]
                else:
                    full_url = None
                    quarter_urls = urls

                _use_cc = role in _chat_completion_roles
                if _use_cc:
                    _log.info("role %s → /v1/chat/completions backend (server-side jinja)", role)

                if has_full and quarter_urls:
                    def _port_of(u: str) -> int:
                        return int(u.rsplit(":", 1)[-1]) if ":" in u else 0

                    def _mk_backend(u: str):
                        return CachingBackend(
                            LlamaServerBackend(ServerConfig(
                                base_url=u, num_slots=num_slots,
                                use_chat_completions=_use_cc,
                            )),
                            _router_for(u),
                        )

                    full_port = _port_of(full_url)

                    # DISPATCH-A2: resolve each endpoint's TRUE topology index by
                    # PORT against NUMA_CONFIG. When the endpoint wired as `full:`
                    # is NOT the topology idx-0 port (a quarter impersonating the
                    # 96-core full — e.g. a quarters-only serving stack whose first
                    # port got the `full:` marker), DEMOTE it into the quarters
                    # pool at its true index rather than stranding it. This (a)
                    # makes all live quarters dispatchable (restores the N-way
                    # ceiling) and (b) makes every quarter's region lock match the
                    # server's physical cores (kills the q0-lock-on-q1-cores
                    # cross-role collision hazard). Only demote when the true index
                    # is resolvable; otherwise keep today's behavior (the
                    # ConcurrencyAwareBackend alignment guard suppresses the full).
                    from src.runtime.instance_topology import (
                        full_instance_port,
                        topology_idx_for_port,
                    )
                    idx0_port = full_instance_port(topology_role)
                    demoted_idx = (
                        topology_idx_for_port(topology_role, full_port)
                        if (idx0_port is not None and full_port != idx0_port)
                        else None
                    )

                    if demoted_idx is not None:
                        quarter_url_list = [full_url] + list(quarter_urls)
                        full_backend = None
                        ca_full_port = 0
                        _log.warning(
                            "role %s: endpoint %s wired as full: is NOT the "
                            "NUMA_CONFIG idx-0 port (%s) for topology role %s — "
                            "DEMOTED into the quarters pool at topology idx %d so "
                            "all %d quarters are dispatchable and each region lock "
                            "matches its physical cores. No full instance served.",
                            role, full_url, idx0_port, topology_role,
                            demoted_idx, len(quarter_url_list),
                        )
                    else:
                        quarter_url_list = list(quarter_urls)
                        full_backend = _mk_backend(full_url)
                        ca_full_port = full_port

                    quarter_backends = []
                    quarter_topology_idxs: list[int] = []
                    for pos, url in enumerate(quarter_url_list):
                        quarter_backends.append(_mk_backend(url))
                        t_idx = topology_idx_for_port(topology_role, _port_of(url))
                        quarter_topology_idxs.append(t_idx if t_idx is not None else pos + 1)

                    self._backends[role] = ConcurrencyAwareBackend(
                        full_backend, quarter_backends,
                        role=role,
                        full_port=ca_full_port,
                        topology_role=topology_role,
                        quarter_topology_idxs=quarter_topology_idxs,
                        native_batch_width=_certified_native_batch_width(
                            self,
                            topology_role=topology_role,
                            full_url=full_url if full_backend is not None else None,
                        ),
                    )
                    if topology_role != role:
                        _log.info(
                            "Concurrency-aware backend for %s uses topology/lock role %s",
                            role, topology_role,
                        )
                    _log.info(
                        "Concurrency-aware backend for %s: %d full + %d quarters (topo idxs %s)",
                        role, 1 if full_backend is not None else 0,
                        len(quarter_backends), quarter_topology_idxs,
                    )
                elif len(quarter_urls) > 1:
                    # Multi-instance role without full: round-robin
                    backends = []
                    for url in quarter_urls:
                        config = ServerConfig(
                            base_url=url, num_slots=num_slots,
                            use_chat_completions=_use_cc,
                        )
                        backend = LlamaServerBackend(config)
                        backends.append(CachingBackend(backend, _router_for(url)))
                    self._backends[role] = RoundRobinBackend(backends, role=role)
                    _log.info("Round-robin backend for %s: %d instances", role, len(quarter_urls))
                else:
                    # Single-instance role
                    url = quarter_urls[0] if quarter_urls else url_str
                    config = ServerConfig(
                        base_url=url, num_slots=num_slots,
                        use_chat_completions=_use_cc,
                    )
                    backend = LlamaServerBackend(config)
                    self._backends[role] = CachingBackend(backend, _router_for(url))

        except ImportError as e:
            _log.warning("CachingBackend not available: %s. Using legacy mode.", e)

    def _init_fleet_backends(
        self,
        server_urls: dict[str, str],
        num_slots: int,
        chat_completion_roles: set[str],
    ) -> frozenset[str]:
        """WP-12: build ONE backend per physical fleet; bind roles to it.

        Returns the set of ``server_urls`` keys the fleet path handled (the
        legacy per-role loop skips them). Fail-closed contract: ANY failure —
        fleet build incoherence, parity violation, unexpected error — returns
        an empty set with a CRITICAL log, leaving the legacy per-role build as
        the serving path for every role.

        For handled roles, ``server_urls[role]`` is rewritten in place to the
        fleet's realized URL value so every downstream primary-URL consumer
        (admission, tap metadata) reads the same fact the dispatcher serves.
        """
        try:
            from src.fleet import _canonical_role, get_fleets_and_bindings, resolve_binding
            from src.backends.llama_server import LlamaServerBackend, ServerConfig
            from src.backends.concurrency_aware import ConcurrencyAwareBackend
            from src.prefix_cache import CachingBackend, PrefixRouter

            if getattr(self, "server_urls_source", "config") != "config":
                _log.info(
                    "fleet layer: request-specific server_urls — keeping the "
                    "legacy per-role build (caller overrides are authoritative)"
                )
                return frozenset()

            state = get_fleets_and_bindings()
            if state is None:
                return frozenset()
            fleets, bindings = state

            role_fleet: dict[str, str] = {}
            for role in server_urls:
                binding = resolve_binding(role, bindings)
                if binding is not None and binding.fleet_id in fleets:
                    role_fleet[role] = binding.fleet_id

            health_tracker = getattr(self, "health_tracker", None)

            def _mk(url: str, use_cc: bool) -> Any:
                return CachingBackend(
                    LlamaServerBackend(ServerConfig(
                        base_url=url, num_slots=num_slots,
                        use_chat_completions=use_cc,
                    )),
                    PrefixRouter(num_slots=num_slots),
                )

            fleet_backend: dict[str, Any] = {}
            for fleet_id in sorted(set(role_fleet.values())):
                fleet = fleets[fleet_id]
                roles_on = sorted(r for r, f in role_fleet.items() if f == fleet_id)
                # use_chat_completions is ROLE-layer policy but is baked into
                # the shared physical backend's ServerConfig — bake the fleet
                # consensus, and FAIL CLOSED (legacy per-role build for this
                # fleet) when bound roles disagree rather than silently
                # mis-routing one of them.
                cc_flags = {
                    r: (r in chat_completion_roles
                        or _canonical_role(r) in chat_completion_roles)
                    for r in roles_on
                }
                if len(set(cc_flags.values())) > 1:
                    _log.critical(
                        "fleet %s: bound roles disagree on /v1/chat/completions "
                        "membership (%s) — fail closed: these roles keep the "
                        "legacy per-role build",
                        fleet_id, cc_flags,
                    )
                    for r in roles_on:
                        role_fleet.pop(r, None)
                    continue
                use_cc = next(iter(cc_flags.values()), False)
                if use_cc:
                    _log.info(
                        "fleet %s → /v1/chat/completions backend (server-side "
                        "jinja) for roles %s", fleet_id, roles_on,
                    )

                full_ep = fleet.full_endpoint
                quarter_eps = fleet.quarter_endpoints
                if len(fleet.endpoints) == 1:
                    backend = _mk(fleet.endpoints[0].url, use_cc)
                else:
                    quarter_topology_idxs = [
                        ep.topology_idx if ep.topology_idx is not None else pos + 1
                        for pos, ep in enumerate(quarter_eps)
                    ]
                    backend = ConcurrencyAwareBackend(
                        _mk(full_ep.url, use_cc) if full_ep is not None else None,
                        [_mk(ep.url, use_cc) for ep in quarter_eps],
                        role=fleet_id,
                        full_port=full_ep.port if full_ep is not None else 0,
                        topology_role=fleet.topology_role,
                        quarter_topology_idxs=quarter_topology_idxs,
                        health_tracker=health_tracker,
                        native_batch_width=_certified_native_batch_width(
                            self,
                            topology_role=fleet.topology_role,
                            full_url=full_ep.url if full_ep is not None else None,
                        ),
                    )
                fleet_backend[fleet_id] = backend
                _log.info(
                    "Fleet backend %s: mode=%s %d full + %d quarters (ports %s) "
                    "serving roles %s%s",
                    fleet_id, fleet.mode,
                    1 if full_ep is not None else 0, len(quarter_eps),
                    list(fleet.ports), roles_on,
                    " [DEGRADED literal]" if fleet.degraded else "",
                )

            handled: set[str] = set()
            for role, fleet_id in role_fleet.items():
                if fleet_id not in fleet_backend:
                    continue
                fleet = fleets[fleet_id]
                # §3 runtime invariant diagnostic: a role's config URL copy
                # disagreeing with the realized fleet is the WP-13 drift class.
                # The fleet is authoritative; the drift is surfaced, not obeyed.
                cfg_ports = _url_str_ports(server_urls.get(role, ""))
                if cfg_ports and set(cfg_ports) != set(fleet.ports):
                    _log.warning(
                        "fleet %s: role %s config URL copy names ports %s but "
                        "the realized fleet serves %s — fleet wins (per-role "
                        "copy drift, WP-13 class)",
                        fleet_id, role, sorted(set(cfg_ports)), sorted(fleet.ports),
                    )
                self._backends[role] = fleet_backend[fleet_id]
                server_urls[role] = fleet.url_value
                handled.add(role)
            return frozenset(handled)
        except Exception:
            _log.critical(
                "WP-12 fleet backend build FAILED — falling back to the legacy "
                "per-role build for every role",
                exc_info=True,
            )
            return frozenset()

    def get_backend(self, role: str) -> Any | None:
        """Get the CachingBackend for a role.

        Args:
            role: Role name (e.g., "worker", "coder", "frontdoor").

        Returns:
            CachingBackend instance or None if not configured.
        """
        return self._backends.get(role)

    def get_cache_stats(self) -> dict[str, dict[str, Any]]:
        """Get cache statistics for all backends.

        Returns:
            Dict mapping role to cache stats dict.
        """
        stats = {}
        for role, backend in self._backends.items():
            if hasattr(backend, "get_stats"):
                stats[role] = backend.get_stats()
        return stats
