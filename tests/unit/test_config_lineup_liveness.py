"""ESC-8 Fix 5: producer-lineup liveness validation in src.config.models.

A producer-derived server lineup (env-filter branch OR runtime-facts branch)
must be validated against the live fleet before it is trusted. A lineup whose
quarterable host-role ports (frontdoor 8070 / worker_general 8072 /
ingest_long_context 8085) are all dead — the env=full / valid-full-manifest
poison signature on a sub-full-only fleet — is rejected and resolution falls
through to the next producer (ultimately stack priors). A TOTAL cold start
(nothing listening anywhere) is NOT the poison class: the lineup is accepted
so the launcher's own cold-start intent survives.

All liveness is injected/mocked: these tests never open a real socket.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from src.config import models

# Dead-in-quarter full host ports; anything else is a live quarter/single-role port.
_FULL_HOST_PORTS = {8070, 8072, 8085}


def _quarters_live(port: int) -> bool:
    """Quarters-only fleet: the full host ports are dead, everything else live."""
    return port not in _FULL_HOST_PORTS


def _fulls_live(_port: int) -> bool:
    """Full fleet up: every probed port (incl. 8070/8072/8085) is listening."""
    return True


def _all_dead(_port: int) -> bool:
    return False


def _server(port: int, *roles: str) -> dict:
    return {"port": port, "roles": list(roles)}


def _runtime_facts(mode: str, servers: list[dict]) -> dict:
    return {
        "schema": "epyc.orchestrator.runtime_facts",
        "schema_version": 1,
        "runtime_stack": {
            "stack_numa_mode": mode,
            "selected_ports": [server["port"] for server in servers],
            "selected_servers": servers,
        },
    }


# --------------------------------------------------------------------------- #
# _selected_servers_are_live                                                   #
# --------------------------------------------------------------------------- #


def test_are_live_rejects_dead_full_host_ports() -> None:
    """env=full-style lineup naming only the dead full host ports is rejected.

    A realistic full lineup also names the live single-instance roles
    (architect_general :8083), so the fleet IS listening and the dead-host-ports
    poison signature is vetoed.
    """
    lineup = [
        _server(8070, "frontdoor"),
        _server(8072, "worker_general"),
        _server(8085, "ingest_long_context"),
        _server(8083, "architect_general"),
    ]
    assert models._selected_servers_are_live(lineup, probe=_quarters_live) is False
    assert models._selected_servers_are_live(lineup, probe=_fulls_live) is True


def test_are_live_accepts_live_quarter_host_ports() -> None:
    lineup = [
        _server(8082, "worker_general"),
        _server(8182, "worker_general"),
        _server(8080, "frontdoor"),
    ]
    assert models._selected_servers_are_live(lineup, probe=_quarters_live) is True


def test_are_live_empty_lineup_is_never_trusted() -> None:
    assert models._selected_servers_are_live(None, probe=_fulls_live) is False
    assert models._selected_servers_are_live([], probe=_fulls_live) is False


def test_are_live_accepts_total_cold_start() -> None:
    """A lineup of only dead ports is ACCEPTED when NOTHING is listening anywhere.

    Total cold start (fleet booting): every producer lineup names
    not-yet-listening ports, so the lineup cannot be a poison signature — it is
    accepted rather than vetoed. The veto requires a live fleet to contradict.
    """
    lineup = [
        _server(8070, "frontdoor"),
        _server(8072, "worker_general"),
        _server(8085, "ingest_long_context"),
    ]
    assert models._selected_servers_are_live(lineup, probe=_all_dead) is True


def test_are_live_no_host_discriminator_cold_start_is_accepted() -> None:
    """Without any quarterable host role, nothing live anywhere = cold start, and
    the lineup is accepted (there is no fleet to poison against)."""
    only_dead = [_server(9999, "worker_fast")]
    assert models._selected_servers_are_live(only_dead, probe=_all_dead) is True
    one_live = [_server(8102, "worker_fast")]
    assert models._selected_servers_are_live(one_live, probe=lambda p: p == 8102) is True


# --------------------------------------------------------------------------- #
# _runtime_or_env_selected_servers — producer precedence + validation          #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _isolate_probe(monkeypatch):
    """Never touch a real socket; default to 'all dead' unless a test overrides."""
    monkeypatch.setattr(models, "_port_listening", _all_dead)
    models._PORT_LISTENING_CACHE.clear()
    yield
    models._PORT_LISTENING_CACHE.clear()


def _patch_runtime_facts(monkeypatch, value):
    import scripts.server.runtime_facts_manifest as rfm

    monkeypatch.setattr(rfm, "read_runtime_stack_selected_servers", lambda **_kw: value)


def test_env_full_with_only_quarters_live_falls_through(monkeypatch, caplog) -> None:
    """env=full + quarters-only fleet → env lineup rejected → None (priors)."""
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
    monkeypatch.setattr(models, "_port_listening", _quarters_live)
    _patch_runtime_facts(monkeypatch, None)

    with caplog.at_level("WARNING"):
        result = models._runtime_or_env_selected_servers()

    assert result is None
    assert any("lineup rejected" in rec.message for rec in caplog.records)


def test_env_full_with_full_ports_live_resolves_full(monkeypatch) -> None:
    """env=full + full ports live → the full manifest lineup is returned."""
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
    monkeypatch.setattr(models, "_port_listening", _fulls_live)
    _patch_runtime_facts(monkeypatch, None)

    result = models._runtime_or_env_selected_servers()

    assert result is not None
    ports = {s.get("port") for s in result}
    # The frontdoor full host port must be present in a resolved full lineup.
    assert 8070 in ports


def test_env_unset_manifest_dead_ports_rejected(tmp_path, monkeypatch, caplog) -> None:
    """env unset + runtime-facts naming dead host ports → rejected → None.

    A realistic full-mode manifest also carries live single-instance roles
    (architect_general :8083), so at least one lineup port IS listening — the
    fleet is up — and the dead-host-ports poison signature is still vetoed.
    """
    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    # Keep the generic bootstrap reader from consuming this host's facts; this
    # test exercises the legacy runtime-manifest producer below.
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(models, "_port_listening", _quarters_live)
    _patch_runtime_facts(
        monkeypatch,
        [
            _server(8070, "frontdoor"),
            _server(8072, "worker_general"),
            _server(8085, "ingest_long_context"),
            _server(8083, "architect_general"),
        ],
    )

    with caplog.at_level("WARNING"):
        result = models._runtime_or_env_selected_servers()

    assert result is None
    assert any("runtime-facts lineup rejected" in rec.message for rec in caplog.records)


def test_env_unset_manifest_all_dead_is_cold_start_accepted(tmp_path, monkeypatch) -> None:
    """env unset + runtime-facts naming dead host ports, NOTHING listening
    anywhere → the manifest lineup is the launcher's own cold-start intent and
    is accepted (not vetoed), instead of falling through to stale priors."""
    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(models, "_port_listening", _all_dead)
    manifest = [
        _server(8070, "frontdoor"),
        _server(8072, "worker_general"),
        _server(8085, "ingest_long_context"),
    ]
    _patch_runtime_facts(monkeypatch, manifest)

    result = models._runtime_or_env_selected_servers()

    assert result == manifest


def test_env_full_cold_start_accepts_env_lineup(monkeypatch) -> None:
    """env=full + NOTHING listening anywhere (total cold start) → the env lineup
    is accepted, not vetoed: no live fleet exists to contradict it."""
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
    monkeypatch.setattr(models, "_port_listening", _all_dead)
    _patch_runtime_facts(monkeypatch, None)

    result = models._runtime_or_env_selected_servers()

    assert result is not None
    ports = {s.get("port") for s in result}
    assert 8070 in ports


def test_env_unset_manifest_live_ports_returned(tmp_path, monkeypatch) -> None:
    """env unset + runtime-facts naming live quarter host ports → returned."""
    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(models, "_port_listening", _quarters_live)
    live_lineup = [
        _server(8082, "worker_general"),
        _server(8080, "frontdoor"),
        _server(8185, "ingest_long_context"),
    ]
    _patch_runtime_facts(monkeypatch, live_lineup)

    result = models._runtime_or_env_selected_servers()

    assert result == live_lineup


def test_env_branch_importerror_is_logged_not_swallowed(monkeypatch, caplog) -> None:
    """Fix 5(b): the env-filter ImportError is logged at WARNING, not swallowed."""
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
    monkeypatch.setattr(models, "_port_listening", _quarters_live)
    _patch_runtime_facts(monkeypatch, None)
    # Force the env-branch `from scripts.server.stack_manifest import ...` to fail
    # exactly as it does under the uvicorn circular-import chain.
    empty = types.ModuleType("scripts.server.stack_manifest")
    monkeypatch.setitem(sys.modules, "scripts.server.stack_manifest", empty)

    with caplog.at_level("WARNING"):
        result = models._runtime_or_env_selected_servers()

    assert result is None
    assert any("env-filter branch failed" in rec.message for rec in caplog.records)


def test_bootstrap_runtime_facts_accepts_quarter_without_scripts_server_imports(
    tmp_path, monkeypatch
) -> None:
    servers = [
        _server(8080, "frontdoor"),
        _server(8082, "worker_general"),
        _server(8185, "ingest_long_context"),
    ]
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "quarter")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(models, "_bootstrap_port_listening", _quarters_live)
    (tmp_path / "orchestrator_runtime_facts.json").write_text(
        json.dumps(_runtime_facts("quarter", servers)),
        encoding="utf-8",
    )

    assert models._runtime_or_env_selected_servers() == servers


@pytest.mark.parametrize(
    ("mode", "servers"),
    [
        (
            "full",
            [
                _server(8070, "frontdoor"),
                _server(8072, "worker_general"),
                _server(8085, "ingest_long_context"),
            ],
        ),
        (
            "both",
            [
                _server(8070, "frontdoor"),
                _server(8080, "frontdoor"),
                _server(8072, "worker_general"),
                _server(8082, "worker_general"),
                _server(8085, "ingest_long_context"),
                _server(8185, "ingest_long_context"),
            ],
        ),
    ],
)
def test_bootstrap_runtime_facts_preserves_realized_full_and_both_modes(
    tmp_path, monkeypatch, mode, servers
) -> None:
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", mode)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(models, "_bootstrap_port_listening", _fulls_live)
    # A partially initialized scripts.server module must not affect the
    # stdlib-only bootstrap reader.
    monkeypatch.setitem(
        sys.modules, "scripts.server.runtime_facts_manifest", types.ModuleType("partial")
    )
    (tmp_path / "orchestrator_runtime_facts.json").write_text(
        json.dumps(_runtime_facts(mode, servers)), encoding="utf-8"
    )

    assert models._bootstrap_runtime_selected_servers() == servers


def test_bootstrap_runtime_facts_rejects_stale_or_dead_facts(tmp_path, monkeypatch) -> None:
    servers = [
        _server(8080, "frontdoor"),
        _server(8082, "worker_general"),
        _server(8185, "ingest_long_context"),
    ]
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "quarter")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    facts = tmp_path / "orchestrator_runtime_facts.json"
    facts.write_text(json.dumps(_runtime_facts("quarter", servers)), encoding="utf-8")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    state = log_dir / "orchestrator_state.json"
    state.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LOG_DIR", str(log_dir))
    os.utime(facts, (100, 100))
    os.utime(state, (101, 101))
    assert models._bootstrap_runtime_selected_servers() is None

    # Make facts fresh, then prove the rejection is caused by the bare TCP
    # liveness check rather than the stale-artifact guard above.
    os.utime(facts, (102, 102))
    probed: list[int] = []

    def dead_probe(port: int) -> bool:
        probed.append(port)
        return _all_dead(port)

    monkeypatch.setattr(models, "_bootstrap_port_listening", dead_probe)
    assert models._bootstrap_runtime_selected_servers() is None
    assert len(probed) == 1
    assert probed[0] in {8080, 8082, 8185}


def test_runtime_selected_aliases_do_not_fall_back_to_dead_static_ports() -> None:
    """Every same-process alias is completed from its host's REALIZED lineup.

    The alias -> host pairs are read from ``_RUNTIME_SELECTED_ROLE_ALIASES``
    itself rather than restated, so a re-pointed alias (the 2026-08-01 W1
    cutover moved coder_escalation frontdoor -> architect_general) is covered
    automatically instead of leaving this guard asserting a dead static port.
    """
    aliases = models._RUNTIME_SELECTED_ROLE_ALIASES
    host_ports = {
        "frontdoor": [8080],
        "worker_general": [8082, 8182],
        # 2026-09-27 ARCHITECT SWAP: the :8083 27B host is architect_critic.
        "architect_critic": [8083],
        "worker_vision": [8086],
    }
    # Every host named by the alias table must be represented in the fixture —
    # otherwise a newly added alias would be silently untested.
    assert set(aliases.values()) <= set(host_ports)

    lineup = [_server(port, host) for host, ports in host_ports.items() for port in ports]
    urls = models._selected_server_url_values(lineup)

    for host, ports in host_ports.items():
        assert urls[host] == ",".join(f"http://localhost:{port}" for port in ports)

    for alias, host in aliases.items():
        assert urls[alias] == urls[host], f"{alias} did not follow its host {host}"

    # Teeth: no alias may resolve to a static full port absent from the lineup.
    for value in urls.values():
        for dead_port in _FULL_HOST_PORTS:
            assert str(dead_port) not in value


def _quarterable_host_and_alias() -> tuple[str, int, list[int], str]:
    """(host role, full port, sibling ports, one alias on it) — DERIVED.

    The fixture below used to name worker_general and the ports 8072/8082/8182 as
    literals. The 2026-09-22 cutover moved the worker lane onto frontdoor's process
    and left worker_general with no NUMA_CONFIG entry at all, so the fixture described
    a fleet the topology no longer declares — `_selected_server_url_values` could not
    identify a full instance for a `topology_role` it does not know, and silently
    dropped the `full:` marker the test exists to protect. Re-pinning to today's ports
    would guarantee the same breakage at the next lineup change.
    """
    from scripts.server import stack_manifest
    from scripts.server.stack_numa import NUMA_CONFIG

    for role in sorted(NUMA_CONFIG):
        cfg = NUMA_CONFIG[role]
        instances = cfg.get("instances") or ()
        full_idx = cfg.get("full_instance_idx")
        if len(instances) < 2 or full_idx is None:
            continue
        meta = stack_manifest.ROLE_LAUNCH_META.get(role) or {}
        aliases = [a for a in (meta.get("shared_with_first_n") or []) if isinstance(a, str)]
        if not aliases:
            continue
        return (
            role,
            instances[full_idx][1],
            [inst[1] for idx, inst in enumerate(instances) if idx != full_idx],
            aliases[0],
        )
    raise AssertionError(
        "no multi-instance host role with an alias in the topology — this fixture "
        "has nothing to exercise and would pass vacuously"
    )


def test_runtime_selected_worker_shape_preserves_full_marker() -> None:
    host, full_port, sibling_ports, alias = _quarterable_host_and_alias()
    # The host answers on every instance of its own fleet; the alias here rides all
    # of them, which is the shape whose `full:` marker is under test. Roles are
    # listed per row the way the launcher emits them, not keyed off topology_role.
    lineup = [
        {
            "port": port,
            "roles": [host, alias],
            "numa_instance": idx,
            "topology_role": host,
        }
        for idx, port in enumerate([full_port, *sibling_ports])
    ]

    urls = models._selected_server_url_values(lineup)

    expected = "full:" + ",".join(
        f"http://localhost:{port}" for port in [full_port, *sibling_ports]
    )
    assert urls[host] == expected
    # The alias spans every instance, so it must carry the same marked fleet: the
    # `full:` prefix is what arms ConcurrencyAwareBackend's full slot, and losing it
    # silently demotes the whole fleet to an unmarked URL list.
    assert urls[alias] == expected
