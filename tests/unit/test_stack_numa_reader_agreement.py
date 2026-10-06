"""Reader-agreement tests for stack NUMA mode projections."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from scripts.server.stack_numa_mode import (
    DASHBOARD_RUNTIME_FALLBACK_NUMA_MODE,
    DEFAULT_STACK_NUMA_MODE,
    env_stack_numa_mode,
    normalize_stack_numa_mode,
)


def _dashboard_ports(mode: str) -> dict[str, list[int]]:
    from src.api.routes import dashboard_topology

    out: dict[str, set[int]] = defaultdict(set)
    for service in dashboard_topology.expected_stack_services(mode):
        port = service.get("port")
        roles = service.get("roles") or []
        if not isinstance(port, int):
            continue
        for role in roles:
            out[str(role)].add(port)
    return {role: sorted(ports) for role, ports in out.items()}


def _stack_prior_ports(mode: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, list[int]]:
    from src.registry.stack_priors import _stack_manifest_info

    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", mode)
    _aliases, roles = _stack_manifest_info()
    return {
        role: sorted(record.get("ports") or [])
        for role, record in roles.items()
        if record.get("ports")
    }


def _alias_hosts(mode: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """DERIVE alias -> host from the same producer the readers use.

    This was a hardcoded {"coder_escalation": "frontdoor", ...} literal. W1
    repointed coder_escalation at architect_general, and the literal went stale
    while still looking authoritative — the test failed asserting [8083] == [8070]
    against a mapping the registry had already changed. `_stack_manifest_info`
    returns this map as its first element and the test was discarding it.

    Deriving it also means a NEWLY declared alias is covered automatically,
    instead of being silently untested until someone remembers to extend a list.
    """
    from src.registry.stack_priors import _stack_manifest_info

    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", mode)
    aliases, _roles = _stack_manifest_info()
    return {str(a): str(h) for a, h in (aliases or {}).items()}


def _host_roles(mode: str, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """DERIVE the host/primary role set from the same producer: every role with a
    port fleet that is NOT an alias.

    This was a hardcoded {"frontdoor", "worker_general", "ingest_long_context",
    "vision_escalation"} literal, and three of its four entries had become aliases.
    The 2026-09-22 cutover made worker_general (with worker_explore / worker_math /
    toolrunner / worker / worker_summarize) an alias on frontdoor's :8070 process,
    so the literal asserted HOST agreement about an ALIAS — and the readers then
    disagreed exactly as they are designed to: the launch views report the instances
    an alias is TAGGED onto (stack_manifest.py, shared_with_first_n_count), the
    serving view reports the whole host fleet the alias is actually served from
    (WP-13 fleet convergence, stack_priors.py ~1241, whose comment records that
    emitting only the tagged ports serialized worker_math eval traffic on one
    quarter). [8080] vs [8080, 8180] is that divergence, asserted by the sibling
    test below as CORRECT. ingest_long_context and vision_escalation are aliases
    too, and passed only vacuously: their GPU hosts run one instance in every mode.

    Deriving the set also covers every NEW host role automatically, and is strictly
    wider than the literal it replaces (15 roles, of which the literal named 1).
    """
    from src.registry.stack_priors import _stack_manifest_info

    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", mode)
    aliases, roles = _stack_manifest_info()
    alias_names = {str(a) for a in (aliases or {})}
    return {
        str(role)
        for role, record in (roles or {}).items()
        if record.get("ports") and str(role) not in alias_names
    }


def _stack_change_guard_ports(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[int]]:
    from scripts.validate import stack_change_guard
    from scripts.validate.stack_change_guard import _launch_manifest_targets

    # Pin the realized-fleet seam to "no signal" so the env var under test
    # governs the launch view; on a live-fleet host the real probe would
    # otherwise override the parameterized mode (ESC-8/WP-13 guard fix).
    monkeypatch.setattr(stack_change_guard, "_realized_launch_numa_mode", lambda: None)
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", mode)
    return {
        role: sorted(target.get("ports") or [])
        for role, target in _launch_manifest_targets().items()
        if target.get("ports")
    }


def _manifest_ports(mode: str) -> dict[str, list[int]]:
    from scripts.server.stack_manifest import HOT_SERVERS, WARM_SERVERS, _filter_by_numa_mode

    out: dict[str, set[int]] = defaultdict(set)
    for server in _filter_by_numa_mode(HOT_SERVERS + WARM_SERVERS, mode):
        port = server.get("port")
        roles = server.get("roles") or []
        if not isinstance(port, int):
            continue
        for role in roles:
            out[str(role)].add(port)
    return {role: sorted(ports) for role, ports in out.items()}


def test_stack_numa_mode_defaults_are_named() -> None:
    assert DEFAULT_STACK_NUMA_MODE == "full"
    assert DASHBOARD_RUNTIME_FALLBACK_NUMA_MODE == "both"
    assert normalize_stack_numa_mode(None) == "full"
    assert normalize_stack_numa_mode("stale", default="both") == "both"
    assert env_stack_numa_mode(environ={}) == "full"
    assert (
        env_stack_numa_mode(
            default="both",
            environ={"ORCHESTRATOR_STACK_NUMA_MODE": "QUARTER"},
        )
        == "quarter"
    )


@pytest.mark.parametrize("mode", ["full", "quarter", "both"])
@pytest.mark.usefixtures("_declared_backend_metadata")
def test_stack_numa_readers_agree_on_host_role_ports(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host/primary roles: the serving view (stack priors) and every
    launch-topology view (raw manifest, dashboard, stack-change guard) agree
    exactly across NUMA modes."""
    stack_prior_ports = _stack_prior_ports(mode, monkeypatch)
    dashboard_ports = _dashboard_ports(mode)
    guard_ports = _stack_change_guard_ports(mode, monkeypatch)
    manifest_ports = _manifest_ports(mode)

    host_roles = _host_roles(mode, monkeypatch)
    assert host_roles, (
        "no host roles derived — the mapping producer returned nothing, so this "
        "agreement check would pass vacuously"
    )
    for role in host_roles:
        assert manifest_ports.get(role) == stack_prior_ports.get(role), role
        assert dashboard_ports.get(role) == stack_prior_ports.get(role), role
        assert guard_ports.get(role) == stack_prior_ports.get(role), role


@pytest.mark.parametrize("mode", ["full", "quarter", "both"])
@pytest.mark.usefixtures("_declared_backend_metadata")
def test_alias_serving_fleet_converges_on_host_launch_views_diverge(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP-13: alias roles (shared_with_first_n) diverge by design.

    The launch-topology views (raw manifest, dashboard, stack-change guard) still
    report only the instances the alias is tagged onto (shared_with_first_n_count),
    and continue to agree with each other. The serving/routing view (stack priors)
    now inherits the host's FULL fleet so it converges with the operative-layer
    Fix-A delegated URL (worker_math/coder_escalation -> host fleet). In
    single-instance ``full`` mode the two coincide; in quarter/both the serving
    fleet is a strict superset of the launch-tagged subset.
    """
    stack_prior_ports = _stack_prior_ports(mode, monkeypatch)
    dashboard_ports = _dashboard_ports(mode)
    guard_ports = _stack_change_guard_ports(mode, monkeypatch)
    manifest_ports = _manifest_ports(mode)

    alias_hosts = _alias_hosts(mode, monkeypatch)
    assert alias_hosts, "no aliases derived; the mapping producer returned nothing"
    for alias, host in alias_hosts.items():
        # Launch-topology views still agree with each other on the tagged subset.
        assert manifest_ports.get(alias) == dashboard_ports.get(alias)
        assert manifest_ports.get(alias) == guard_ports.get(alias)
        # Fleet convergence: the alias serving fleet equals its host serving fleet.
        assert stack_prior_ports.get(alias) == stack_prior_ports.get(host)
        # And the serving fleet is a superset of the launch-tagged subset.
        tagged = set(manifest_ports.get(alias) or [])
        serving = set(stack_prior_ports.get(alias) or [])
        assert tagged <= serving

        # Strictness is keyed on whether the HOST fans out, not on the mode.
        # The old form asserted `tagged < serving` for every alias in
        # quarter/both, which was really a proxy for "hosts fan out into
        # quarters". That proxy broke when coder_escalation moved to
        # architect_general: GPU-hosted roles (architect_general :8083,
        # worker_vision :8086) run ONE instance in every mode, so their aliases
        # are tagged onto the whole fleet and the subset can never be strict.
        if len(serving) == 1:
            assert tagged == serving, (
                f"{alias}: single-instance host {host} -> alias must be tagged "
                f"onto its whole fleet"
            )
        else:
            assert tagged < serving, (
                f"{alias}: host {host} fans out to {sorted(serving)} but the "
                f"launch view tags {sorted(tagged)}; WP-13 requires the serving "
                f"view to strictly exceed the launch-tagged subset"
            )


@pytest.fixture
def _declared_backend_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Owned path metadata for pure compilation; no binary or store evidence."""
    import subprocess
    from src.registry import kernel_paths

    directories = {}
    for backend in ("cpu", "gpu"):
        directory = tmp_path / "declared-backends" / backend
        directory.mkdir(mode=0o700, parents=True)
        directories[backend] = directory

    def declared_backend_dir(backend):
        if backend not in directories:
            raise AssertionError(f"unexpected metadata backend: {backend!r}")
        return directories[backend]

    def refuse_child(*args, **kwargs):
        raise AssertionError("pure metadata fixture must not create a child process")

    monkeypatch.setattr(kernel_paths, "backend_dir", declared_backend_dir)
    # Preserve real CPU [] and GPU vendor-path library policy.
    monkeypatch.setattr(subprocess, "Popen", refuse_child)
    yield directories
    for directory in directories.values():
        assert not list(directory.iterdir()), "metadata fixture acquired a binary or output"
