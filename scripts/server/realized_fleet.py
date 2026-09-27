"""Realized-fleet NUMA-mode detection (bare TCP connect, no HTTP).

ESC-8 (2026-07-22): the launcher/writer/compiler must never trust an env or
manifest ``ORCHESTRATOR_STACK_NUMA_MODE`` value that the *running* fleet
contradicts. This module derives the true NUMA mode of the live stack by
partitioning the NUMA_CONFIG ports into the quarterable-host "full" ports
(dead in quarter mode) versus their "quarter" siblings, and probing which of
those ports are actually listening.

Import-safe: stdlib ``socket`` only; ``NUMA_CONFIG`` is imported lazily so this
module can be imported from anywhere in the process tree (and from tests)
without coupling to running infrastructure. Only ``derive_realized_numa_mode``
opens sockets, and only bare TCP connects (``connect_ex``) with a short timeout
against localhost — never an HTTP request to a llama-server.

The three consumers:
  * ``runtime_facts_manifest`` (Fix 1) classifies the *realized state* ports.
  * ``orchestrator_stack.start_orchestrator`` (Fix 3) probes the live fleet to
    refuse exporting a contradicted mode.
  * ``stack_priors`` compile (Fix 6) refuses to default a clean shell to
    "full" and derives/validates against the realized fleet.

The socket ``connect`` function is injectable everywhere so tests can simulate
an arbitrary live-port set without touching real sockets.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterable
from typing import Any

# Localhost-only, short-timeout defaults. A closed localhost port returns
# ECONNREFUSED immediately, so the timeout only bounds filtered/dropped ports;
# the whole probe budget stays well under ~1s even for the full port universe.
DEFAULT_PROBE_HOST = "127.0.0.1"
DEFAULT_PROBE_TIMEOUT_S = 0.15

# Injectable connect predicate: (host, port) -> True when host:port accepts a
# TCP connection. Callers/tests may substitute a pure function.
ConnectFn = Callable[[str, int], bool]


def _numa_config(numa_config: dict[str, Any] | None) -> dict[str, Any]:
    if numa_config is not None:
        return numa_config
    try:
        from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]

        return NUMA_CONFIG
    except Exception:
        return {}


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def full_instance_ports(numa_config: dict[str, Any] | None = None) -> set[int]:
    """Ports of the quarterable roles' full (all-region) instance.

    Only roles with ``full_instance_idx`` AND more than one instance qualify —
    these are exactly the ports that are LIVE in ``full`` mode and DEAD in
    ``quarter`` mode (frontdoor 8070, worker_general 8072, ingest 8085).
    Single-instance roles (architects 8074/8083, vision 8086/8087) are excluded: they
    are present in every mode and are not a full/quarter discriminator.
    """
    out: set[int] = set()
    for cfg in _numa_config(numa_config).values():
        if not isinstance(cfg, dict):
            continue
        instances = cfg.get("instances") or []
        full_idx = cfg.get("full_instance_idx")
        if not isinstance(full_idx, int) or isinstance(full_idx, bool):
            continue
        if len(instances) <= 1 or not (0 <= full_idx < len(instances)):
            continue
        entry = instances[full_idx]
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            port = _int_or_none(entry[1])
            if port is not None:
                out.add(port)
    return out


def quarter_instance_ports(numa_config: dict[str, Any] | None = None) -> set[int]:
    """Ports of the quarterable roles' non-full (quarter) instances."""
    out: set[int] = set()
    for cfg in _numa_config(numa_config).values():
        if not isinstance(cfg, dict):
            continue
        instances = cfg.get("instances") or []
        full_idx = cfg.get("full_instance_idx")
        if not isinstance(full_idx, int) or isinstance(full_idx, bool):
            continue
        if len(instances) <= 1:
            continue
        for idx, entry in enumerate(instances):
            if idx == full_idx:
                continue
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                port = _int_or_none(entry[1])
                if port is not None:
                    out.add(port)
    return out


def classify_numa_mode_from_ports(
    ports: Iterable[int],
    numa_config: dict[str, Any] | None = None,
) -> str | None:
    """Classify a NUMA mode from a set of realized/live ports.

    Returns ``"full"`` / ``"quarter"`` / ``"both"`` when the ports include
    quarterable-role full and/or quarter ports, or ``None`` when the ports carry
    no full/quarter discriminator (empty, or only single-instance roles). A
    ``None`` result is deliberately fail-safe: readers/writers treat it as
    "unknown, do not fabricate" rather than defaulting to ``"full"``.
    """
    cfg = _numa_config(numa_config)
    fulls = full_instance_ports(cfg)
    quarters = quarter_instance_ports(cfg)
    observed = {p for p in (_int_or_none(port) for port in ports) if p is not None}
    has_full = bool(observed & fulls)
    has_quarter = bool(observed & quarters)
    if has_full and has_quarter:
        return "both"
    if has_full:
        return "full"
    if has_quarter:
        return "quarter"
    return None


def _default_connect(host: str, port: int, *, timeout: float) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


def probe_listening(
    ports: Iterable[int],
    *,
    connect: ConnectFn | None = None,
    host: str = DEFAULT_PROBE_HOST,
    timeout: float = DEFAULT_PROBE_TIMEOUT_S,
) -> set[int]:
    """Return the subset of ``ports`` that accept a bare TCP connection."""
    probe = connect or (lambda h, p: _default_connect(h, p, timeout=timeout))
    live: set[int] = set()
    for port in ports:
        p = _int_or_none(port)
        if p is None or p <= 0:
            continue
        try:
            if probe(host, p):
                live.add(p)
        except OSError:
            continue
    return live


def derive_realized_numa_mode(
    *,
    connect: ConnectFn | None = None,
    numa_config: dict[str, Any] | None = None,
    host: str = DEFAULT_PROBE_HOST,
    timeout: float = DEFAULT_PROBE_TIMEOUT_S,
) -> str | None:
    """Probe the live fleet and classify its realized NUMA mode.

    Probes only the quarterable roles' full ∪ quarter ports (bare TCP connect,
    localhost). Returns ``None`` when nothing in that universe is listening
    (stack down / no quarterable roles up) — callers then fall back rather than
    fabricating a mode.
    """
    cfg = _numa_config(numa_config)
    universe = full_instance_ports(cfg) | quarter_instance_ports(cfg)
    if not universe:
        return None
    live = probe_listening(universe, connect=connect, host=host, timeout=timeout)
    return classify_numa_mode_from_ports(live, cfg)
