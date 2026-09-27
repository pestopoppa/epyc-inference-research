"""WP-12 server-fleet layer — one physical description per llama-server fleet.

Design: epyc-root handoffs/active/wp12-fleet-layer-design.md (operator-endorsed
2026-07-22). Governing contract: the operator-ratified fabric constitution
(heterogeneous-slot-fabric-residency.md banner) — fleet/role bindings are POLICY
DATA never code; one fact per physical resource, realized-probed; realized-first
truth over launch intent; fail-closed on anything unverifiable.

A ``ServerFleet`` is the single description of one physical llama-server
deployment, built from the registry ``server_mode`` source of truth (which fleet
exists, which roles ride it) plus generated stack priors (which ports are
realized) plus ``NUMA_CONFIG`` (topology idx / cpuset / placement policy per
port). A ``RoleBinding`` maps a logical role onto exactly one fleet; every
role-layer property (timeout, prompt, sampling, priority) stays looked up by
role name upstream of the physical backend.

The env/runtime-facts URL producers are deliberately NOT consulted here: fleet
identity comes from the registry + priors + NUMA_CONFIG only, which makes the
ESC-8 env-clobber class (``ORCHESTRATOR_STACK_NUMA_MODE=full`` wiring hot roles
to dead full ports) structurally impossible rather than a setdefault-ordering
accident (design §2.1).

Everything in this module is inert unless ``ORCHESTRATOR_FLEET_LAYER=1``
(default off). Flag-off, no production path imports it.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.roles import Role

_log = logging.getLogger(__name__)

FLEET_LAYER_ENV = "ORCHESTRATOR_FLEET_LAYER"


def fleet_layer_enabled() -> bool:
    """Live read of the WP-12 rollout flag (default OFF)."""
    return os.environ.get(FLEET_LAYER_ENV) == "1"


class FleetBuildError(RuntimeError):
    """The fleet layer could not derive a coherent fleet description.

    Callers treat this as fail-closed for the fleet layer: the per-role legacy
    build remains the serving path and the failure is logged CRITICAL.
    """


class FleetParityError(FleetBuildError):
    """A role bound to a fleet resolves to a different endpoint set than the
    fleet itself — the structural replacement for WP-13's parity drift-guard
    test (design §3 invariant). Fail closed: the fleet layer refuses to build.
    """


# Degraded-mode bootstrap (design §8 risk 1): when stack priors are absent
# (fresh clone, pre-launch API) the fleet layer keeps exactly ONE literal
# fallback per FLEET — never per role — so the degraded path still resolves
# without re-introducing per-role copies. Ports are resolved against
# NUMA_CONFIG like any other endpoint set (full/quarter alignment included).
def _derive_degraded_fallback_ports() -> dict[str, tuple[int, ...]]:
    """Degraded-mode ports, DERIVED from NUMA_CONFIG rather than restated.

    2026-07-30. These were literal 5-port tuples per fleet, and they went stale
    the moment the topology changed from 1 full + 4 quarters to 1 full + 2
    halves — advertising 8280/8380/8282/8382/8385/8485, which no longer exist.
    That is not a cosmetic staleness: several dispatch paths fail OPEN on a port
    they cannot resolve to a topology index. ``backend.py`` falls back to a
    positional index, ``placement.py`` treats an empty region set as
    overlap-free, and ``cpu_region_lock`` yields NO LOCK AT ALL for an unknown
    ``(role, idx)``. A stale literal here is therefore a route to unlocked
    dispatch, not merely a wrong number.

    Deriving keeps the module's stated contract — the docstring already claimed
    ports "are resolved against NUMA_CONFIG like any other endpoint set" — and
    removes the second source of truth. If NUMA_CONFIG is unavailable (fresh
    clone, no scripts/ on the path) this yields an empty mapping, and the caller
    already handles that by skipping the fleet and leaving roles on the legacy
    per-role build.
    """
    cfg = _default_numa_config()
    out: dict[str, tuple[int, ...]] = {}
    # 2026-09-27 ARCHITECT SWAP: architect_critic added — it is now the host row of
    # the :8083 27B fleet (coder_escalation + ingest_long_context ride it);
    # architect_general is the :8074 full-CPU row.
    for role in (
        "frontdoor",
        "worker_general",
        "architect_general",
        "architect_critic",
        "ingest_long_context",
    ):
        entry = cfg.get(role) if isinstance(cfg, Mapping) else None
        instances = (entry or {}).get("instances") if isinstance(entry, Mapping) else None
        if not instances:
            continue
        ports: list[int] = []
        for inst in instances:
            try:
                ports.append(int(inst[1]))
            except (TypeError, ValueError, IndexError):
                continue
        if ports:
            out[role] = tuple(ports)
    return out


_FLEET_DEGRADED_FALLBACK_PORTS_CACHE: dict[str, tuple[int, ...]] | None = None


def _fleet_degraded_fallback_ports() -> dict[str, tuple[int, ...]]:
    """Lazily derive and cache. Deliberately NOT a module-level constant.

    ``_default_numa_config`` is defined later in this module, so evaluating at
    import time raises NameError. Deferring also means a caller that never hits
    the degraded path never imports ``stack_numa`` at all.
    """
    global _FLEET_DEGRADED_FALLBACK_PORTS_CACHE
    if _FLEET_DEGRADED_FALLBACK_PORTS_CACHE is None:
        _FLEET_DEGRADED_FALLBACK_PORTS_CACHE = _derive_degraded_fallback_ports()
    return _FLEET_DEGRADED_FALLBACK_PORTS_CACHE


@dataclass(frozen=True)
class FleetEndpoint:
    """One live llama-server endpoint of a fleet.

    ``topology_idx``/``regions`` are resolved by PORT against NUMA_CONFIG
    (DISPATCH-A2 semantics, computed once at fleet build): a quarter mislabeled
    as the full instance by an upstream URL producer cannot exist here, because
    ``is_full`` is derived from the port's true topology index, never from a
    ``full:`` string marker.
    """

    port: int
    url: str
    topology_idx: int | None
    regions: frozenset[str]
    is_full: bool


@dataclass(frozen=True)
class ServerFleet:
    """The single description of one physical llama-server deployment.

    ``fleet_id`` doubles as the topology/lock role for every endpoint — one
    region-lock identity per fleet (design §2.1/§3). ``mode`` is the REALIZED
    serving mode derived from the live endpoint set, not the static launch
    default (kills the WP-14 phantom-lineup class).
    """

    fleet_id: str
    endpoints: tuple[FleetEndpoint, ...]
    mode: str  # "full" | "quarter" | "mixed" | "single" | "unknown"
    placement_policy: str | None
    bound_roles: tuple[str, ...]
    model_binding: str | None
    degraded: bool = False

    @property
    def topology_role(self) -> str:
        """The ONE region-lock identity for every endpoint in the fleet."""
        return self.fleet_id

    @property
    def full_endpoint(self) -> FleetEndpoint | None:
        for ep in self.endpoints:
            if ep.is_full:
                return ep
        return None

    @property
    def quarter_endpoints(self) -> tuple[FleetEndpoint, ...]:
        return tuple(ep for ep in self.endpoints if not ep.is_full)

    @property
    def ports(self) -> tuple[int, ...]:
        return tuple(ep.port for ep in self.endpoints)

    @property
    def endpoint_urls(self) -> tuple[str, ...]:
        return tuple(ep.url for ep in self.endpoints)

    @property
    def url_value(self) -> str:
        """Config-compatible URL string for this fleet.

        The ``full:`` marker is emitted ONLY when a true (port-aligned) full
        endpoint is realized — unlike the stack-priors serializer, which marks
        the first of any multi-port list. A quarters-only fleet therefore never
        advertises a phantom full.
        """
        full = self.full_endpoint
        urls = [ep.url for ep in self.endpoints if not ep.is_full]
        if full is not None:
            return ",".join([f"full:{full.url}"] + urls) if urls else full.url
        return ",".join(urls)


@dataclass(frozen=True)
class RoleBinding:
    """What a role *is* once the physical layer is factored out (design §2.2).

    ``capacity_cap`` and ``placement_policy_override`` are role-layer policy
    data born bounded/reversible per the fabric contract; they default to
    None (= full fleet / fleet default) and are carried as data only — no
    dispatch consumer enforces them yet (enforcement is a gated follow-up,
    not silently invented here).
    """

    role: str
    fleet_id: str
    model_binding: str | None
    capacity_cap: int | None = None
    placement_policy_override: str | None = None


def _canonical_role(name: str) -> str:
    """Canonicalize a config role name to its physical-layer role name.

    Delegates to the config layer's alias table (one fact): ``worker`` →
    ``worker_general``, ``coder`` → ``coder_escalation``, legacy Role aliases
    via ``Role.from_string`` — with the ``worker_coder``/``worker_fast``
    carve-out preserved (those are distinct physical servers, NOT worker
    aliases, despite the legacy Role alias table).
    """
    try:
        from src.config.models import _canonical_server_url_name

        return _canonical_server_url_name(name)
    except Exception:
        role = Role.from_string(name)
        return str(role) if role is not None else name


def _default_priors_path() -> Path:
    from src.config.models import _get_default_stack_priors_path

    return Path(_get_default_stack_priors_path())


def _default_registry_server_mode() -> Mapping[str, Any]:
    import yaml

    from src.config import get_config

    registry_path = Path(get_config().paths.registry_path)
    with registry_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    server_mode = raw.get("server_mode")
    if not isinstance(server_mode, dict) or not server_mode:
        raise FleetBuildError(
            f"registry {registry_path} has no usable server_mode section"
        )
    return server_mode


def _default_numa_config() -> Mapping[str, Any]:
    try:
        from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]

        return NUMA_CONFIG if isinstance(NUMA_CONFIG, dict) else {}
    except Exception:
        return {}


def _priors_ports_by_role(priors_path: Path) -> dict[str, list[int]]:
    """Realized serving ports per live-stack role from generated priors."""
    from src.registry.stack_priors import (
        live_stack_role_records,
        stack_prior_serving,
        stack_prior_serving_ports,
    )

    out: dict[str, list[int]] = {}
    try:
        records = live_stack_role_records(priors_path)
    except Exception as exc:
        _log.warning("fleet build: stack priors unreadable (%s: %s)", type(exc).__name__, exc)
        return {}
    for role, record in records.items():
        ports = stack_prior_serving_ports(stack_prior_serving(record))
        if ports:
            out[role] = ports
    return out


def _build_endpoints(
    fleet_id: str,
    ports: list[int] | tuple[int, ...],
    numa_config: Mapping[str, Any],
) -> tuple[FleetEndpoint, ...]:
    from src.runtime.instance_topology import cpu_list_to_regions, topology_idx_for_port

    ncfg = numa_config.get(fleet_id) if isinstance(numa_config, Mapping) else None
    instances = (ncfg.get("instances") or []) if isinstance(ncfg, Mapping) else []
    full_idx = ncfg.get("full_instance_idx") if isinstance(ncfg, Mapping) else None
    if not isinstance(full_idx, int) or isinstance(full_idx, bool):
        full_idx = None

    endpoints: list[FleetEndpoint] = []
    seen: set[int] = set()
    for port in ports:
        if not isinstance(port, int) or isinstance(port, bool) or port <= 0 or port in seen:
            continue
        seen.add(port)
        topo = topology_idx_for_port(fleet_id, port, dict(numa_config) if numa_config else {})
        regions: frozenset[str] = frozenset()
        if topo is not None and 0 <= topo < len(instances):
            entry = instances[topo]
            if entry:
                regions = cpu_list_to_regions(entry[0])
        endpoints.append(
            FleetEndpoint(
                port=port,
                url=f"http://localhost:{port}",
                topology_idx=topo,
                regions=regions,
                is_full=(topo is not None and full_idx is not None and topo == full_idx),
            )
        )
    # Aligned full first, then quarters in topology order (unknown-idx last, by
    # port, stable) — the dispatcher's expected full-then-quarters layout.
    endpoints.sort(
        key=lambda ep: (
            0 if ep.is_full else 1,
            ep.topology_idx if ep.topology_idx is not None else 1_000_000,
            ep.port,
        )
    )
    return tuple(endpoints)


def _classify_mode(
    fleet_id: str,
    endpoints: tuple[FleetEndpoint, ...],
    numa_config: Mapping[str, Any],
) -> str:
    ncfg = numa_config.get(fleet_id) if isinstance(numa_config, Mapping) else None
    instances = (ncfg.get("instances") or []) if isinstance(ncfg, Mapping) else []
    full_idx = ncfg.get("full_instance_idx") if isinstance(ncfg, Mapping) else None
    quarterable = (
        isinstance(full_idx, int)
        and not isinstance(full_idx, bool)
        and len(instances) > 1
    )
    if not quarterable:
        return "single" if len(endpoints) == 1 else "unknown"
    has_full = any(ep.is_full for ep in endpoints)
    has_quarter = any(not ep.is_full for ep in endpoints)
    if has_full and has_quarter:
        return "mixed"
    if has_full:
        return "full"
    if has_quarter:
        return "quarter"
    return "unknown"


def build_fleets_and_bindings(
    *,
    registry_server_mode: Mapping[str, Any] | None = None,
    numa_config: Mapping[str, Any] | None = None,
    priors_path: Path | None = None,
) -> tuple[dict[str, ServerFleet], dict[str, RoleBinding]]:
    """Build the fleet map + role bindings from the three declarative sources.

    Sources (design §2.1 construction): registry ``server_mode`` rows (fleet
    existence + ``shared_with`` role membership), generated stack priors
    (realized serving ports — the SoT), ``NUMA_CONFIG`` (shapes / topology
    idxs / placement policy, resolved by port). The env and runtime-facts
    producers are never consulted.

    Raises ``FleetParityError``/``FleetBuildError`` on any incoherence — the
    caller falls back to the legacy per-role build (fail closed for the fleet
    layer, loud in the logs).
    """
    server_mode = (
        registry_server_mode if registry_server_mode is not None else _default_registry_server_mode()
    )
    numa = numa_config if numa_config is not None else _default_numa_config()
    priors = _priors_ports_by_role(
        priors_path if priors_path is not None else _default_priors_path()
    )

    fleets: dict[str, ServerFleet] = {}
    bindings: dict[str, RoleBinding] = {}

    # 2026-08-04: `alias_of` is the registry's SECOND way of declaring a role
    # co-hosted on another row's process, and this builder knew only the first
    # (`shared_with`). The W1 cutover expressed coder_escalation -> :8083 as an
    # `alias_of: architect_general` row that keeps its own model/quality facts,
    # while architect_general lists it in `shared_with` — two mutually
    # consistent declarations of ONE physical server. Iterating every row as a
    # fleet built a phantom 'coder_escalation' fleet on :8083 and then tripped
    # the double-binding guard against the real architect_general fleet, so
    # ORCHESTRATOR_FLEET_LAYER=1 could not build at all against the production
    # registry (latent only because the flag defaults off).
    #
    # An alias row is not a physical resource, so it gets no fleet: it is bound
    # onto its host's fleet exactly like a `shared_with` member. The
    # double-binding guard below is untouched and still fires on a genuine
    # conflict (two different hosts claiming one role).
    alias_rows: dict[str, str] = {}
    for key, cfg in server_mode.items():
        if not isinstance(cfg, Mapping):
            continue
        target = cfg.get("alias_of")
        if not isinstance(target, str) or not target:
            continue
        if target not in server_mode:
            raise FleetBuildError(
                f"server_mode row {key!r} declares alias_of={target!r}, which is "
                "not a server_mode row — an alias must name its host fleet"
            )
        alias_id = _canonical_role(str(key))
        host_id = _canonical_role(target)
        if alias_id == host_id:
            raise FleetBuildError(
                f"server_mode row {key!r} declares alias_of={target!r}, which "
                "resolves to itself — an alias cannot host itself"
            )
        alias_rows[alias_id] = host_id

    aliases_by_host: dict[str, list[str]] = {}
    for alias_id, host_id in alias_rows.items():
        aliases_by_host.setdefault(host_id, []).append(alias_id)

    for key, cfg in server_mode.items():
        if not isinstance(cfg, Mapping):
            continue
        fleet_id = _canonical_role(str(key))
        if fleet_id in alias_rows:
            # Logical alias row: no fleet of its own; bound onto its host below.
            continue
        if fleet_id in fleets:
            raise FleetBuildError(
                f"server_mode rows {key!r} and a prior row both resolve to fleet "
                f"{fleet_id!r} — one fleet must be defined exactly once"
            )

        degraded = False
        ports: list[int] | tuple[int, ...] = priors.get(fleet_id, [])
        if not ports:
            ports = _fleet_degraded_fallback_ports().get(fleet_id, ())
            degraded = bool(ports)
            if not ports:
                _log.info(
                    "fleet build: no realized ports and no degraded literal for "
                    "server_mode row %r (fleet %r) — skipping (roles stay on the "
                    "legacy per-role build)",
                    key,
                    fleet_id,
                )
                continue
            _log.warning(
                "fleet build: stack priors carry no ports for fleet %r — using "
                "the degraded per-fleet literal %s",
                fleet_id,
                tuple(ports),
            )

        endpoints = _build_endpoints(fleet_id, list(ports), numa)
        if not endpoints:
            continue

        shared_with = cfg.get("shared_with")
        bound: list[str] = [fleet_id]
        if isinstance(shared_with, list):
            for alias in shared_with:
                if isinstance(alias, str) and alias:
                    canonical = _canonical_role(alias)
                    if canonical not in bound:
                        bound.append(canonical)
        # Rows that named THIS fleet via `alias_of` bind here too.
        for canonical in aliases_by_host.get(fleet_id, ()):
            if canonical not in bound:
                bound.append(canonical)

        model_binding = cfg.get("model_role") or cfg.get("model")
        ncfg = numa.get(fleet_id) if isinstance(numa, Mapping) else None
        placement_policy = (
            ncfg.get("placement_policy") if isinstance(ncfg, Mapping) else None
        )

        fleet = ServerFleet(
            fleet_id=fleet_id,
            endpoints=endpoints,
            mode=_classify_mode(fleet_id, endpoints, numa),
            placement_policy=placement_policy if isinstance(placement_policy, str) else None,
            bound_roles=tuple(bound),
            model_binding=model_binding if isinstance(model_binding, str) else None,
            degraded=degraded,
        )
        fleets[fleet_id] = fleet

        for role in bound:
            existing = bindings.get(role)
            if existing is not None and existing.fleet_id != fleet_id:
                raise FleetBuildError(
                    f"role {role!r} is bound to two fleets ({existing.fleet_id!r} "
                    f"and {fleet_id!r}) — one fact per physical resource"
                )
            bindings[role] = RoleBinding(
                role=role,
                fleet_id=fleet_id,
                model_binding=fleet.model_binding,
            )

    _validate_parity(fleets, bindings, priors)
    return fleets, bindings


def _validate_parity(
    fleets: Mapping[str, ServerFleet],
    bindings: Mapping[str, RoleBinding],
    priors: Mapping[str, list[int]],
) -> None:
    """Design §3 invariant: every bound role resolves to the identical endpoint
    set as its fleet. The priors artifact still carries per-alias records
    (WP-13 inheritance); any divergence there is exactly the drift class the
    fleet layer exists to kill — fail closed instead of picking a side.
    """
    for role, binding in bindings.items():
        fleet = fleets[binding.fleet_id]
        role_ports = priors.get(role)
        if role_ports is None:
            continue
        if sorted(role_ports) != sorted(fleet.ports):
            raise FleetParityError(
                f"role {role!r} resolves to ports {sorted(role_ports)} but its "
                f"fleet {fleet.fleet_id!r} realizes {sorted(fleet.ports)} — "
                f"endpoint-set parity violated (WP-13 drift class)"
            )


# ── Cached process-wide accessor ────────────────────────────────────────────

_CACHE_LOCK = threading.Lock()
_FLEETS_CACHE: tuple[dict[str, ServerFleet], dict[str, RoleBinding]] | None = None
_BUILD_FAILED = False


def get_fleets_and_bindings() -> tuple[dict[str, ServerFleet], dict[str, RoleBinding]] | None:
    """Cached fleet build from the default sources.

    Returns None when the build has failed (logged CRITICAL once) — callers
    must then keep the legacy per-role behavior. ``reset_fleet_cache()``
    clears both the result and the failure latch.
    """
    global _FLEETS_CACHE, _BUILD_FAILED
    with _CACHE_LOCK:
        if _FLEETS_CACHE is not None:
            return _FLEETS_CACHE
        if _BUILD_FAILED:
            return None
        try:
            _FLEETS_CACHE = build_fleets_and_bindings()
        except Exception as exc:
            _BUILD_FAILED = True
            _log.critical(
                "WP-12 fleet layer: fleet build FAILED (%s: %s) — falling back "
                "to the legacy per-role backend build for this process",
                type(exc).__name__,
                exc,
            )
            return None
        return _FLEETS_CACHE


def reset_fleet_cache() -> None:
    """Clear the cached fleet build (tests / config reload)."""
    global _FLEETS_CACHE, _BUILD_FAILED
    with _CACHE_LOCK:
        _FLEETS_CACHE = None
        _BUILD_FAILED = False


def resolve_binding(
    role_name: str,
    bindings: Mapping[str, RoleBinding],
) -> RoleBinding | None:
    """Resolve a raw config role name to its fleet binding (raw, then canonical)."""
    binding = bindings.get(role_name)
    if binding is not None:
        return binding
    return bindings.get(_canonical_role(role_name))


def fleet_id_for_role(role_name: str) -> str | None:
    """Fleet id serving ``role_name``, or None (unbound / build unavailable)."""
    state = get_fleets_and_bindings()
    if state is None:
        return None
    binding = resolve_binding(role_name, state[1])
    return binding.fleet_id if binding is not None else None


def compiled_fleet_fallback_map() -> dict[Role, tuple[Role, ...]] | None:
    """The fallback map with same-fleet edges compiled out (design §4).

    Fallback is meaningful iff it changes the physical fleet: a same-fleet
    candidate retries the identical backend + identical (already-open)
    breaker — the ``forced_role_fallback`` churn class. Cross-fleet edges and
    edges involving unbound roles are kept verbatim. Returns None when the
    fleet build is unavailable (callers keep the legacy map).
    """
    state = get_fleets_and_bindings()
    if state is None:
        return None
    _, bindings = state

    from src.roles import _FALLBACK_MAP

    compiled: dict[Role, tuple[Role, ...]] = {}
    for role, targets in _FALLBACK_MAP.items():
        role_binding = resolve_binding(role.value, bindings)
        kept: list[Role] = []
        for target in targets:
            target_binding = resolve_binding(target.value, bindings)
            if (
                role_binding is not None
                and target_binding is not None
                and role_binding.fleet_id == target_binding.fleet_id
            ):
                _log.debug(
                    "fleet fallback compile: dropping same-fleet edge %s -> %s "
                    "(fleet %s)",
                    role.value,
                    target.value,
                    role_binding.fleet_id,
                )
                continue
            kept.append(target)
        compiled[role] = tuple(kept)
    return compiled
