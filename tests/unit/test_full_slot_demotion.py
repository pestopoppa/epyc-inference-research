"""DISPATCH-A2: misaligned-`full:` demotion into the quarters pool.

When the endpoint wired as `full:` is NOT the topology idx-0 port (a quarter
impersonating the 96-core full — the live frontdoor-fleet wiring), the
construction site demotes it into the quarters pool at its TRUE topology index
(port-resolved) instead of stranding it. This restores the N-way concurrency
ceiling AND makes every quarter's region lock match the server's physical cores
(killing the q0-lock-on-q1-cores cross-role collision hazard).

Uses the REAL topology of the fleet that hosts worker_general (host read from
the registry's server_mode, instances from NUMA_CONFIG — never restated) so the
idx→port→cpuset consistency is asserted against the live truth. Port literals
are deliberately absent: the fleet shape has changed three times (2026-07-23
big+quarters restoration, 2026-07-30 quarters retirement to 1 full + 2 halves,
2026-09-22 lineup cutover making worker_general an ALIAS of frontdoor with no
NUMA_CONFIG entry of its own) and a hardcoded fixture both fails the tests that
assert it and — worse — silently passes the ones that do not, because dispatch
fails OPEN on a port it cannot resolve to a topology index (an unresolvable
endpoint gets an EMPTY region set, so its lock can never conflict). The
2026-09-22 change is what broke collection here (``KeyError: 'worker_general'``
from ``NUMA_CONFIG[WG]``); the fix derives the host instead of naming it.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "server"))

ca_mod = importlib.import_module("src.backends.concurrency_aware")
from src.backends.concurrency_aware import _get_base_url
from src.config.models import ServerURLsConfig, reset_stack_prior_server_url_cache
from src.llm_primitives.backend import BackendMixin
from src.runtime.instance_topology import (
    cpu_list_to_regions,
    get_instance_regions,
    topology_idx_for_port,
)
from scripts.server.stack_numa import NUMA_CONFIG

WG = "worker_general"
WM = "worker_math"
FD = "frontdoor"
CE = "coder_escalation"
WS = "worker_summarize"

# The MI210 27B host of coder_escalation. It was architect_general until the
# 2026-09-27 ARCHITECT SWAP (operator-decided) relabelled that :8083 process
# architect_critic. The name is kept as AG ("architect host") for diff stability.
AG = "architect_critic"


def _topology_ports(role: str) -> tuple[int, list[int]]:
    """(aligned-full port, sibling instance ports) for a role, from NUMA_CONFIG."""
    cfg = NUMA_CONFIG[role]
    instances = cfg["instances"]
    full_idx = cfg["full_instance_idx"]
    return instances[full_idx][1], [
        inst[1] for idx, inst in enumerate(instances) if idx != full_idx
    ]


def _fleet_default(role: str) -> str:
    """The `full:`-prefixed default URL list the topology implies for ``role``."""
    full_port, siblings = _topology_ports(role)
    parts = [f"full:http://localhost:{full_port}"]
    parts += [f"http://localhost:{p}" for p in siblings]
    return ",".join(parts)


def _registry_server_mode() -> dict:
    import yaml

    return yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text(encoding="utf-8")
    )["server_mode"]


def _registry_alias_to_host() -> dict[str, str]:
    """alias role -> hosting server_mode row, DERIVED from the registry.

    Two declaration forms carry the same meaning and both must be honoured:
    `shared_with` (co-hosted roles on one process) and an `alias_of` row (a role
    that keeps its own registry facts but launches no server of its own — how
    the 2026-08-01 W1 cutover expressed coder_escalation -> architect_general).
    """
    server_mode = _registry_server_mode()
    alias_to_host: dict[str, str] = {}
    for host, row in server_mode.items():
        for shared in row.get("shared_with") or []:
            alias_to_host[str(shared)] = host
    for role, row in server_mode.items():
        target = row.get("alias_of")
        if target:
            alias_to_host[role] = str(target)
    return alias_to_host


def _registry_aliases_of(host: str) -> list[str]:
    return sorted(a for a, h in _registry_alias_to_host().items() if h == host)


# The physical fleet serving worker_general (and worker_math, which shares it).
# worker_general has NO NUMA_CONFIG entry of its own since the 2026-09-22 lineup
# cutover; its topology — ports, instance idxs, lock regions — is its HOST's.
WG_HOST = _registry_alias_to_host().get(WG, WG)

HOST_FULL, HOST_SIBLINGS = _topology_ports(WG_HOST)
FD_FULL, FD_SIBLINGS = _topology_ports(FD)
HOST_SIBLING_IDXS = [topology_idx_for_port(WG_HOST, p) for p in HOST_SIBLINGS]
FD_SIBLING_IDXS = [topology_idx_for_port(FD, p) for p in FD_SIBLINGS]

# The canonical worker_general default (its host's aligned idx-0 full + every
# sibling instance). worker_math shares that physical server, so its default URL
# list must carry the SAME shape or its ConcurrencyAwareBackend serializes on a
# single instance (live EV-11c incident: ~3 q/min instead of the fanned-out ~7).
_WG_DEFAULT = _fleet_default(WG_HOST)

# The canonical frontdoor default (aligned idx-0 full + every sibling instance).
# The frontdoor-fleet aliases the registry declares via
# server_mode.frontdoor.shared_with are served by the frontdoor GGUF
# (Qwen3.6-35B-A3B Q8, shared mmap). Fix A delegates their URL default to
# frontdoor so each carries this SAME shape and its ConcurrencyAwareBackend fans
# out under topology_role=frontdoor instead of serializing on the single 8070
# port those roles were previously pinned to.
_FD_DEFAULT = _fleet_default(FD)


class _Host(BackendMixin):
    """Minimal BackendMixin host to exercise _init_caching_backends directly."""

    def __init__(self) -> None:
        self._backends: dict = {}


def _build(server_urls: dict, num_slots: int = 1) -> dict:
    host = _Host()
    host._init_caching_backends(server_urls, num_slots)
    return host._backends


def _port(backend) -> int | None:
    url = _get_base_url(backend)
    return int(url.rsplit(":", 1)[-1]) if url and ":" in url else None


def _urls(full_port: int, quarter_ports: tuple[int, ...]) -> str:
    parts = [f"full:http://localhost:{full_port}"]
    parts += [f"http://localhost:{p}" for p in quarter_ports]
    return ",".join(parts)


class _LockCtx:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    def __enter__(self):
        if not self.ok:
            from src.runtime.cpu_region_lock import CpuRegionLockTimeout

            raise CpuRegionLockTimeout("mock")
        return ["/tmp/cpu_region.mock.lock"]

    def __exit__(self, *exc):
        return False


# ── construction: misaligned full is DEMOTED, not stranded ───────────────────

def _demoted_worker_urls() -> str:
    """`full:` marks the FIRST sibling instance — a half impersonating the full."""
    return _urls(HOST_SIBLINGS[0], tuple(HOST_SIBLINGS[1:]))


def test_misaligned_full_demoted_into_quarters_pool() -> None:
    """Live shape: `full:` marks a sibling instance, not the aligned idx-0 full.
    It is demoted → no full served, EVERY sibling dispatchable at its true idx."""
    backends = _build({WG_HOST: _demoted_worker_urls()})
    be = backends[WG_HOST]
    assert isinstance(be, ca_mod.ConcurrencyAwareBackend)
    assert be._full is None                       # misaligned full demoted → none served
    # N-way ceiling restored: every realized sibling instance, none stranded.
    assert len(be._quarters) == len(HOST_SIBLINGS)
    assert be._quarter_topology_idx == HOST_SIBLING_IDXS
    assert [_port(q) for q in be._quarters] == HOST_SIBLINGS
    # Every idx is a REAL topology index — a stale port would resolve to None
    # and get a positional idx with an empty (never-conflicting) region set.
    assert all(idx is not None for idx in be._quarter_topology_idx)
    assert be._full_slot_aligned is True          # no full slot → vacuously aligned
    assert (
        be.max_concurrency() >= len(HOST_SIBLINGS)
        if hasattr(be, "max_concurrency")
        else True
    )


def test_demoted_region_locks_match_physical_cores() -> None:
    """idx → port → cpuset consistency (the anti-shift invariant): the region the
    dispatcher LOCKS for each quarter equals the physical cpuset of the server at
    that port, per NUMA_CONFIG."""
    backends = _build({WG_HOST: _demoted_worker_urls()})
    be = backends[WG_HOST]
    ir = get_instance_regions()
    port_to_cpulist = {int(e[1]): e[0] for e in NUMA_CONFIG[WG_HOST]["instances"]}

    for i, q in enumerate(be._quarters):
        topo = be._quarter_topology_idx[i]
        port = _port(q)
        locked_regions = ir[(WG_HOST, topo)]                 # what the lock covers
        physical_regions = cpu_list_to_regions(port_to_cpulist[port])  # server's real cores
        assert locked_regions == physical_regions, (
            f"quarter {i} (port {port}) locks {sorted(locked_regions)} "
            f"but physically runs on {sorted(physical_regions)}"
        )
        assert topology_idx_for_port(WG_HOST, port) == topo


def test_demoted_endpoint_dispatchable_and_locks_true_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The demoted endpoint (first sibling → its true topo idx) is reachable, and the
    dispatcher locks the topology index matching the chosen backend's port."""
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    monkeypatch.setenv("ORCHESTRATOR_PLACEMENT_STATE_MACHINE", "1")
    backends = _build({WG_HOST: _demoted_worker_urls()})
    be = backends[WG_HOST]

    attempted: list[int] = []

    def _mock_lock(role, instance_idx, timeout_s=None, deadline_s=None):
        attempted.append(instance_idx)
        return _LockCtx(ok=True)

    monkeypatch.setattr("src.runtime.cpu_region_lock.cpu_region_lock_for_instance", _mock_lock)
    monkeypatch.setattr("src.runtime.cpu_region_lock.active_region_holders", lambda *a, **k: {})
    monkeypatch.setattr("src.runtime.cpu_region_lock.held_regions_by_role", lambda *a, **k: {})

    with be._dispatch(session_id="d0") as (_bk, idx, is_full):
        assert is_full is False
        assert 0 <= idx < len(HOST_SIBLINGS)
        # The locked topology idx is the chosen quarter's TRUE (port-resolved) idx,
        # never the all-region idx 0.
        assert attempted[-1] == be._quarter_topology_idx[idx]
        assert attempted[-1] in HOST_SIBLING_IDXS
        assert 0 not in attempted
    # The demoted endpoint occupies internal slot 0 at its own topology idx.
    assert be._quarter_topology_idx[0] == HOST_SIBLING_IDXS[0]
    assert _port(be._quarters[0]) == HOST_SIBLINGS[0]


# ── construction: a REAL full (port == idx-0) is preserved unchanged ──────────

def test_aligned_full_preserved() -> None:
    """When `full:` IS the topology idx-0 port (a real 96-core full deployed),
    the full slot is served exactly as before and siblings keep their true idxs."""
    backends = _build({WG_HOST: _urls(HOST_FULL, tuple(HOST_SIBLINGS))})
    be = backends[WG_HOST]
    assert be._full is not None                    # real full served
    assert be._full_port == HOST_FULL
    assert be._full_slot_aligned is True
    assert len(be._quarters) == len(HOST_SIBLINGS)
    assert be._quarter_topology_idx == HOST_SIBLING_IDXS
    assert [_port(q) for q in be._quarters] == HOST_SIBLINGS


def test_aligned_full_emits_full_candidate_on_solo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real full + a SOLO_PREFER_FULL role → solo dispatch routes to the full
    instance (idx 0), proving the explicit-topology-idx change did not disturb
    the aligned full path. (Synthetic 1 full + 4 quarters frontdoor shape via
    direct construction, with regions and locks mocked.)"""
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    monkeypatch.setenv("ORCHESTRATOR_PLACEMENT_STATE_MACHINE", "1")

    class _Stub:
        def __init__(self, url):
            self.config = type("C", (), {"base_url": url})()

    full = _Stub("http://localhost:8070")
    quarters = [_Stub(f"http://localhost:80{80 + i * 100}") for i in range(4)]
    be = ca_mod.ConcurrencyAwareBackend(
        full_backend=full, quarter_backends=quarters,
        role="frontdoor", full_port=8070,  # aligned idx-0 for frontdoor
    )
    assert be._full_slot_aligned is True
    assert be._quarter_topology_idx == [1, 2, 3, 4]  # legacy positional default

    attempted: list[int] = []

    def _mock_lock(role, instance_idx, timeout_s=None, deadline_s=None):
        attempted.append(instance_idx)
        return _LockCtx(ok=True)

    _FRONTDOOR_REGIONS = {
        ("frontdoor", 0): frozenset({"q0", "q1"}),
        ("frontdoor", 1): frozenset({"q0"}),
        ("frontdoor", 2): frozenset({"q1"}),
        ("frontdoor", 3): frozenset({"q2"}),
        ("frontdoor", 4): frozenset({"q3"}),
    }
    monkeypatch.setattr("src.runtime.cpu_region_lock.cpu_region_lock_for_instance", _mock_lock)
    monkeypatch.setattr("src.runtime.cpu_region_lock.active_region_holders", lambda *a, **k: {})
    monkeypatch.setattr("src.runtime.cpu_region_lock.held_regions_by_role", lambda *a, **k: {})
    monkeypatch.setattr(
        "src.runtime.instance_topology.get_instance_regions",
        lambda: dict(_FRONTDOOR_REGIONS),
    )

    with be._dispatch(session_id="solo") as (_bk, idx, is_full):
        assert is_full is True     # full instance chosen on solo
        assert idx == -1
        assert attempted[-1] == 0  # topology idx 0 (the real full) locked


# ── worker_math shares worker_general's host fleet: default URL parity ─────
#
# worker_math has NO NUMA_CONFIG entry of its own; it dispatches on the same
# physical server as worker_general (since the 2026-09-22 lineup cutover both
# are declared in server_mode.frontdoor.shared_with — the Qwen3.6-35B-A3B
# frontdoor fleet; before it, the gemma `worker` server). Its default URL list
# must therefore carry that host's FULL shape (aligned full + every sibling) so
# its ConcurrencyAwareBackend fans out instead of serializing on a single
# instance (EV-11c live incident: the worker_math arm ran ~3 q/min instead of
# the fanned-out ~7 because its default carried only ONE quarter, 8082).


def _fallback_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Force the legacy literal fallbacks: missing stack priors + runtime facts
    ignored, so ServerURLsConfig reads _LEGACY_SERVER_URL_FALLBACKS
    deterministically (same isolation pattern as tests/unit/test_config.py)."""
    monkeypatch.setenv("ORCHESTRATOR_IGNORE_RUNTIME_STACK_FACTS", "1")
    monkeypatch.setenv(
        "ORCHESTRATOR_PATHS_STACK_PRIORS_PATH", str(tmp_path / "missing.yaml")
    )
    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    reset_stack_prior_server_url_cache()


def test_worker_math_default_url_list_matches_worker_general_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(a)+(c): worker_math's default URL list = the host's aligned idx-0 full
    plus EVERY sibling instance the topology declares, byte-for-byte identical
    to worker_general's (and the host's own) default. The shape is derived from
    the registry + NUMA_CONFIG, so a topology change updates the expectation
    while a config-only drift (the EV-11c failure: worker_math narrowed to one
    endpoint) still fails here."""
    _fallback_env(monkeypatch, tmp_path)
    try:
        cfg = ServerURLsConfig()
        # The fan-out this guard exists to protect must be real, not a 1-endpoint
        # fleet that would make the parity assertions vacuous.
        assert len(HOST_SIBLINGS) >= 2
        # (c) worker_general default matches its host's realized topology.
        assert cfg.worker_general == _WG_DEFAULT
        assert cfg.worker_general == getattr(cfg, WG_HOST)
        # (a) worker_math yields the same full + siblings shape.
        assert cfg.worker_math == _WG_DEFAULT
        assert cfg.worker_math == cfg.worker_general
        parts = cfg.worker_math.split(",")
        assert parts[0] == f"full:http://localhost:{HOST_FULL}"   # aligned idx-0 full
        assert parts[1:] == [f"http://localhost:{p}" for p in HOST_SIBLINGS]
    finally:
        reset_stack_prior_server_url_cache()


def test_worker_math_backend_builds_siblings_under_host_topology(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(b): the shipped worker_math / worker_general defaults construct CA
    backends whose topology/lock role is the HOST fleet, with the aligned idx-0
    full served (not demoted) and EVERY sibling instance at its TRUE
    (port-resolved) topology idx. Regression: the old single-quarter default
    built only ONE quarter and serialized dispatch."""
    _fallback_env(monkeypatch, tmp_path)
    try:
        cfg = ServerURLsConfig()
        # The host MUST be co-present so the aliases' topology role is
        # resolvable by matching (full-stripped) URL lists — as it always is in
        # the live server_urls map.
        backends = _build(
            {WG_HOST: getattr(cfg, WG_HOST), WG: cfg.worker_general, WM: cfg.worker_math}
        )
    finally:
        reset_stack_prior_server_url_cache()

    assert len(HOST_SIBLINGS) >= 2
    for role in (WM, WG):
        be = backends[role]
        assert isinstance(be, ca_mod.ConcurrencyAwareBackend), role
        # Topology/lock role is the physical host fleet, so region locks
        # collide correctly with it instead of a phantom empty per-alias
        # topology.
        assert be._topology_role == WG_HOST, role
        assert be._role == role
        # Aligned full (host idx-0 port) → served, not demoted.
        assert be._full is not None, role
        assert be._full_port == HOST_FULL, role
        assert be._full_slot_aligned is True, role
        # Every sibling at its TRUE (port-resolved) topology idx — the
        # anti-serialization tooth is that the pool width equals the topology's
        # sibling count, never 1.
        assert len(be._quarters) == len(HOST_SIBLINGS), role
        assert be._quarter_topology_idx == HOST_SIBLING_IDXS, role
        assert [_port(q) for q in be._quarters] == HOST_SIBLINGS, role
        for topo, port in zip(be._quarter_topology_idx, HOST_SIBLINGS):
            assert topology_idx_for_port(WG_HOST, port) == topo

    # The host itself: same aligned full + sibling-pool shape.
    host_be = backends[WG_HOST]
    assert isinstance(host_be, ca_mod.ConcurrencyAwareBackend)
    assert host_be._topology_role == WG_HOST
    assert host_be._quarter_topology_idx == HOST_SIBLING_IDXS


def test_shared_worker_fleet_url_defaults_do_not_drift() -> None:
    """DRIFT GUARD: worker_general, worker_math and every other role the registry
    declares on the same host fleet that ALSO carries its OWN literal URL default
    must keep that literal identical to the host's. Interim guard until backends
    are derived from server_mode directly — a future edit to one but not the
    other fails here, naming the shared-fleet relationship and the
    denormalization site."""
    from src.config.models import _LEGACY_SERVER_URL_FALLBACKS as FB

    alias_to_host = _registry_alias_to_host()
    # The registry links the guard depends on must stay intact.
    assert alias_to_host.get(WM) == WG_HOST, (
        f"worker_math is no longer declared on {WG_HOST!r} (worker_general's "
        f"host) — got {alias_to_host.get(WM)!r}; the two no longer share a fleet "
        "and this guard must be re-derived."
    )
    # The legacy `worker` server_mode row canonicalizes to worker_general in the
    # fleet builder, so it must name the same host or the builder double-binds.
    if "worker" in alias_to_host:
        assert alias_to_host["worker"] == WG_HOST

    shared = _registry_aliases_of(WG_HOST)
    # Parity set = roles on the host fleet that ALSO denormalize their own
    # literal default (worker_explore / worker have none, so they cannot drift).
    parity_roles = [r for r in shared if r in FB]
    for anchor in (WG, WM):
        assert anchor in parity_roles, (
            f"{anchor} lost its own default in _LEGACY_SERVER_URL_FALLBACKS (or its "
            "registry declaration); the drift guard is now vacuous — restore the "
            "literal or update the guard."
        )
    host_default = FB[WG_HOST]
    for role in parity_roles:
        assert FB[role] == host_default, (
            f"shared-fleet URL drift: role {role!r} shares the {WG_HOST!r} server "
            f"(server_mode.{WG_HOST}.shared_with / alias_of in "
            f"orchestration/model_registry.yaml) but its default URL list in "
            f"_LEGACY_SERVER_URL_FALLBACKS (src/config/models.py) diverges from "
            f"the host fleet:\n"
            f"    {role}: {FB[role]!r}\n"
            f"    {WG_HOST}: {host_default!r}\n"
            f"Shared-fleet roles MUST carry an identical URL list (edit BOTH or "
            f"neither) until backends are derived from server_mode directly."
        )


# ── Fix A: frontdoor-fleet aliases inherit frontdoor's whole fleet ────────────
#
# Fix A's rule is "an alias role delegates its URL field to its HOST fleet".
# Before it, such roles resolved their OWN name, got a single port, and their
# ConcurrencyAwareBackend serialized on ONE endpoint under a phantom empty
# topology. The rule is invariant; the HOST membership is not, so the alias set
# is read from server_mode.frontdoor.shared_with rather than restated:
#   - worker_summarize is declared in server_mode.frontdoor.shared_with.
#   - `coder` is a candidate-role LABEL with no server_mode row of its own; it
#     stays on frontdoor (hop 1 of escalation_chains.coder).
#   - coder_escalation LEFT this fleet at the 2026-08-01 W1 cutover
#     (`alias_of: architect_general`): it is now one MI210 process, so Fix A
#     correctly gives it that single endpoint — covered below as its own case.


def test_frontdoor_shared_fields_match_frontdoor_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(a): every role the REGISTRY declares as a frontdoor-fleet alias delegates
    its URL default to frontdoor and equals frontdoor's default byte-for-byte;
    a role the registry has moved off the fleet follows its NEW host instead."""
    _fallback_env(monkeypatch, tmp_path)
    try:
        cfg = ServerURLsConfig()
        fd_aliases = _registry_aliases_of(FD)
        # Non-vacuity: the frontdoor fleet must still declare at least one alias,
        # otherwise this parity loop would pass by having nothing to check.
        assert fd_aliases, (
            "server_mode.frontdoor.shared_with is empty — the Fix A parity guard "
            "has no alias left to protect; re-point it at the declared host."
        )
        # frontdoor default matches the realized topology.
        assert cfg.frontdoor == _FD_DEFAULT
        parts = cfg.frontdoor.split(",")
        assert parts[0] == f"full:http://localhost:{FD_FULL}"   # aligned idx-0 full
        assert parts[1:] == [f"http://localhost:{p}" for p in FD_SIBLINGS]

        # Each declared frontdoor alias inherits the SAME full + siblings shape.
        for alias in fd_aliases:
            assert getattr(cfg, alias) == _FD_DEFAULT, alias
            assert getattr(cfg, alias) == cfg.frontdoor, alias
        # `coder` is a candidate-role label with no server_mode row; it stays on
        # frontdoor by design (escalation_chains.coder hop 1).
        assert cfg.coder == cfg.frontdoor

        # W1 cutover: coder_escalation is `alias_of: architect_general`, so Fix A
        # points it at THAT host's endpoint — not frontdoor's fleet, whose ports
        # do not serve its model.
        assert _registry_alias_to_host()[CE] == AG
        assert cfg.coder_escalation == getattr(cfg, AG)
        assert cfg.coder_escalation != cfg.frontdoor
    finally:
        reset_stack_prior_server_url_cache()


def test_frontdoor_shared_backends_build_four_quarters_under_frontdoor_topology(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(b): the shipped default of every REGISTRY-declared frontdoor alias builds
    a CA backend whose topology/lock role is frontdoor, with the aligned idx-0
    full served (not demoted) and every frontdoor sibling at its TRUE
    (port-resolved) topology idx. Regression: the old single-8070 default built
    ONE quarter and serialized dispatch under a phantom empty topology.

    coder_escalation left this fleet at the W1 cutover and is covered here as
    its own case: it must share architect_general's single MI210 endpoint."""
    _fallback_env(monkeypatch, tmp_path)
    fd_aliases = _registry_aliases_of(FD)
    assert fd_aliases, "no frontdoor-fleet alias left to exercise"
    try:
        cfg = ServerURLsConfig()
        # frontdoor MUST be co-present so the aliases' topology role is resolvable
        # by matching (full-stripped) URL lists.
        role_urls = {FD: cfg.frontdoor, AG: getattr(cfg, AG), CE: cfg.coder_escalation}
        role_urls.update({alias: getattr(cfg, alias) for alias in fd_aliases})
        backends = _build(role_urls)
    finally:
        reset_stack_prior_server_url_cache()

    for alias in fd_aliases:
        be = backends[alias]
        assert isinstance(be, ca_mod.ConcurrencyAwareBackend), alias
        # Topology/lock role aliases onto frontdoor (shared physical fleet), so
        # region locks collide correctly with frontdoor instead of a phantom
        # empty per-alias topology.
        assert be._topology_role == FD, alias
        assert be._role == alias
        # Aligned full (frontdoor idx-0 port) → served, not demoted.
        assert be._full is not None, alias
        assert be._full_port == FD_FULL, alias
        assert be._full_slot_aligned is True, alias
        # Every frontdoor sibling at its TRUE (port-resolved) topology idx.
        assert len(be._quarters) == len(FD_SIBLINGS), alias
        assert be._quarter_topology_idx == FD_SIBLING_IDXS, alias
        assert [_port(q) for q in be._quarters] == FD_SIBLINGS, alias
        for topo, port in zip(be._quarter_topology_idx, FD_SIBLINGS):
            assert topology_idx_for_port(FD, port) == topo

    # frontdoor itself unchanged: same aligned full + sibling-pool shape.
    fd_be = backends[FD]
    assert isinstance(fd_be, ca_mod.ConcurrencyAwareBackend)
    assert fd_be._topology_role == FD
    assert fd_be._full_port == FD_FULL
    assert len(FD_SIBLINGS) >= 2
    assert len(fd_be._quarters) == len(FD_SIBLINGS)
    assert fd_be._quarter_topology_idx == FD_SIBLING_IDXS

    # W1 cutover coverage: coder_escalation delegates to architect_general, a
    # single GPU-host-lane process. Fix A's rule ("delegate to your HOST fleet")
    # is unchanged; the host is. One endpoint here is the host's real width, not
    # the pre-Fix-A phantom-topology serialization.
    assert _registry_alias_to_host()[CE] == AG
    ce_be = backends[CE]
    assert not isinstance(ce_be, ca_mod.ConcurrencyAwareBackend)
    assert _get_base_url(ce_be) == _get_base_url(backends[AG])
    assert _port(ce_be) == _port(backends[AG])


def test_frontdoor_shared_fleet_url_defaults_do_not_drift() -> None:
    """DRIFT GUARD (all declared aliases): every role the registry declares as an
    alias — via `shared_with` OR via an `alias_of` row — that ALSO carries its own
    literal in _LEGACY_SERVER_URL_FALLBACKS must keep that literal identical to
    its HOST's literal.

    The alias set is DERIVED from orchestration/model_registry.yaml, not restated.
    The previous version pinned `sorted(shared_with) == [coder_escalation,
    worker_summarize]` and told the reader to "update this guard to the new
    declared reality" when it fired; it fired at the 2026-08-01 W1 cutover
    ("frontdoor sheds coder_escalation, keeps worker_summarize"), which is exactly
    the update — expressed as the derivation the literal was standing in for.
    Deriving also gives the guard MORE reach than the pin did: the pinned version
    exercised an EMPTY parity set for the frontdoor fleet (neither declared alias
    carried an FB literal), whereas the alias_of arm now actively covers
    coder_escalation -> architect_general."""
    from src.config.models import _LEGACY_SERVER_URL_FALLBACKS as FB

    server_mode = _registry_server_mode()
    alias_to_host = _registry_alias_to_host()

    def _fb_key(row_name: str) -> str | None:
        """Map a server_mode ROW to the role name its FB literal is filed under.

        Most rows are named for their serving role; `worker` is filed under its
        model_role (`worker_general`), which is the same indirection the
        worker-fleet guard above depends on.
        """
        if row_name in FB:
            return row_name
        model_role = (server_mode.get(row_name) or {}).get("model_role")
        return model_role if model_role in FB else None

    # Non-vacuity: at least one declared alias must actually be parity-checked,
    # otherwise a guard that checks nothing would report green forever.
    checked: list[str] = []
    for alias, host in sorted(alias_to_host.items()):
        host_key = _fb_key(host)
        if alias not in FB or host_key is None:
            continue
        checked.append(alias)
        assert FB[alias] == FB[host_key], (
            f"alias URL drift: role {alias!r} is declared an alias of {host!r} in "
            f"orchestration/model_registry.yaml (shared_with / alias_of) but its "
            f"default URL list in _LEGACY_SERVER_URL_FALLBACKS "
            f"(src/config/models.py) diverges from its host's:\n"
            f"    {alias}: {FB[alias]!r}\n"
            f"    {host_key}: {FB[host_key]!r}\n"
            f"Alias roles MUST carry an identical URL list (edit BOTH or neither) "
            f"until backends are derived from server_mode directly. See "
            f"docs/runbooks/role-alias-change-runbook.md."
        )
    assert checked, (
        "no declared alias carries its own _LEGACY_SERVER_URL_FALLBACKS literal — "
        "this drift guard has become vacuous; re-derive it from the registry."
    )
    # Anchor: the W1 cutover pair must be inside the checked set, so a future
    # edit cannot quietly drop the alias whose host most recently moved.
    assert CE in checked and alias_to_host[CE] == AG
