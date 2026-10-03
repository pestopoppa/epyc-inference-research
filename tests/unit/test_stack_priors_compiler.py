"""Tests for derived stack-prior compilation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.registry.stack_priors import (
    STACK_PRIORS_VERSION,
    StackPriorsCompileError,
    StackPriorsModeError,
    _realized_compile_numa_mode,
    canonical_stack_role_id,
    compile_stack_priors,
    live_role_primary_ports,
    _launch_runtime_record,
    live_stack_lock_role_sets,
    live_stack_role_ids,
    live_stack_role_records,
    live_stack_safe_non_stream_roles,
    live_stack_serving_slot_limits,
    live_stack_serving_url_values,
    live_stack_slot_query_ports,
    live_warm_worker_slots,
    load_stack_priors_artifact,
    _policy_hints,
    stack_prior_endpoint_port,
    stack_prior_launch_entries,
    stack_prior_launch_modes,
    stack_prior_model_mem_gb,
    stack_prior_primary_port,
    stack_prior_serving_url_value,
    stack_prior_serving_ports,
    stack_prior_uses_shared_worker_launch,
    _launch_record,
    _serving_record,
    _stack_manifest_info,
    _server_mode_launch_requirement_overrides,
    validate_stack_priors_contract,
)

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Lineup derivation helpers (SSU-F5)
# ---------------------------------------------------------------------------
# THE DEFECT THESE CLOSE (session friction audit 2026-09-22 §2, *restated
# derivation*): a fact that is a function of ``server_mode.<host>.shared_with``
# was copied into these fixtures as a PORT LITERAL. The 2026-09-22 lineup change
# moved worker_general / worker_explore / worker_math / toolrunner off their own
# :8072 server onto frontdoor's :8070 process, and eleven tests in this file went
# red for a reason unrelated to anything they assert. Re-pinning 8072 -> 8070
# would make today green and guarantee the identical breakage at the next lineup
# change -- that is the defect re-applied, not the fix.
#
# So every port expectation below is RECOMPUTED from the same declarations the
# launcher reads, then diffed against the compiler's projection:
#
#   launch_manifest.yaml  port_map          -> stack_manifest.PORT_MAP
#   launch_manifest.yaml  role_launch_meta  -> stack_manifest.ROLE_LAUNCH_META
#                                              (its ``shared_with_first_n`` is
#                                              itself DERIVED from the registry's
#                                              ``server_mode.<host>.shared_with``)
#   stack_topology.yaml   numa_config       -> stack_numa.NUMA_CONFIG
#
# They read the LIVE module attributes rather than a snapshot, so they follow a
# real lineup change AND the monkeypatched fleets some tests below install.
#
# When a literal genuinely IS the thing under test (a deliberate historical pin),
# it is kept and the comment says WHY, so the next reader does not "modernise" it.


def _lineup_alias_map() -> dict[str, str]:
    """alias -> host, recomputed from the launcher's own declaration."""
    from scripts.server import stack_manifest

    out: dict[str, str] = {}
    for host, meta in stack_manifest.ROLE_LAUNCH_META.items():
        if not isinstance(meta, dict):
            continue
        for alias in meta.get("shared_with_first_n") or []:
            if isinstance(alias, str):
                out[alias] = str(host)
    return out


def _lineup_host_of(role: str) -> str:
    """The role whose llama-server process ``role`` answers on (itself if host)."""
    return _lineup_alias_map().get(role, role)


def _lineup_is_alias(role: str) -> bool:
    return role in _lineup_alias_map()


def _lineup_launch_mode(role: str) -> str | None:
    """The launch mode of the process ``role`` rides (``worker_pool``, ...)."""
    from scripts.server import stack_manifest

    meta = stack_manifest.ROLE_LAUNCH_META.get(_lineup_host_of(role))
    return (meta or {}).get("mode") if isinstance(meta, dict) else None


def _declared_fleet_ports(role: str, numa_mode: str = "full") -> list[int]:
    """The serving fleet ``role`` must resolve to, from the DECLARATIONS.

    Derived from stack_topology's ``numa_config`` (instances + full_instance_idx)
    for the resolved HOST, falling back to launch_manifest's ``port_map`` for a
    role with no NUMA wiring. Deliberately computed from the declaration rather
    than from ``stack_manifest.HOT_SERVERS``/``WARM_SERVERS``, which is the
    structure the code under test consumes -- a recompute, not a second copy.

    ``numa_mode`` follows ``stack_manifest._filter_by_numa_mode``: ``full`` keeps
    only the full instance, ``both`` keeps the whole fleet, anything else (the
    legacy ``quarter`` token) keeps the sub-full siblings.
    """
    from scripts.server import stack_manifest
    from scripts.server.stack_numa import NUMA_CONFIG

    host = _lineup_host_of(role)
    cfg = NUMA_CONFIG.get(host) or {}
    instances = list(cfg.get("instances") or [])
    full_idx = cfg.get("full_instance_idx")

    if len(instances) <= 1 or not isinstance(full_idx, int):
        ports = [inst[1] for inst in instances]
        if not ports:
            declared = stack_manifest.PORT_MAP.get(host)
            ports = [declared] if isinstance(declared, int) else []
        return sorted({p for p in ports if isinstance(p, int)})

    if numa_mode == "both":
        keep = range(len(instances))
    elif numa_mode == "full":
        keep = [full_idx]
    else:
        keep = [i for i in range(len(instances)) if i != full_idx]
    return sorted({instances[i][1] for i in keep})


def _declared_primary_port(role: str, numa_mode: str = "full") -> int | None:
    ports = _declared_fleet_ports(role, numa_mode)
    return ports[0] if ports else None


def _declared_url(role: str, numa_mode: str = "full") -> str | None:
    port = _declared_primary_port(role, numa_mode)
    return f"http://localhost:{port}" if isinstance(port, int) else None



def test_stack_manifest_info_defaults_to_launcher_full_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)

    _aliases, roles = _stack_manifest_info()

    # SSU-F5: every port here is DERIVED (see the helpers above). The literals
    # that stood in their place were restatements of `shared_with` + `port_map`
    # and went red at the 2026-09-22 lineup change; these follow it. Each role is
    # asserted against BOTH its declared fleet and the host it resolves to, so an
    # alias that silently stops riding its host is still caught.
    for role in (
        "frontdoor",
        "coder_escalation",
        "worker_summarize",
        "worker_general",
        "ingest_long_context",
        "vision_escalation",
    ):
        assert roles[role]["ports"] == _declared_fleet_ports(role), role
        assert roles[role]["url"] == _declared_url(role), role
        assert roles[role]["ports"] == roles[_lineup_host_of(role)]["ports"], role
    assert (
        roles["vision_escalation"]["launch"]["requirements"]
        == roles["worker_vision"]["launch"]["requirements"]
    )
    # 2026-08-01 W1 cutover: was "Qwen2.5-VL-7B-Instruct".
    assert "Qwen3-VL-30B-A3B-Instruct" in roles["vision_escalation"]["launch"][
        "requirements"
    ]["model_path"]


def test_stack_manifest_info_can_compile_explicit_both_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "both")

    _aliases, roles = _stack_manifest_info()

    # SSU-F5: DERIVED from numa_config (full instance + sub-full siblings) rather
    # than pinned. The literals here have already been rewritten twice by topology
    # events -- the 2026-07-30 quarter retirement ([8070, 8080, 8180, 8280, 8380]
    # -> full + 2 halves) and the 2026-09-22 lineup change -- which is the whole
    # argument for computing them.
    for role in ("frontdoor", "worker_general"):
        assert roles[role]["ports"] == _declared_fleet_ports(role, "both"), role
        assert roles[role]["url"] == _declared_url(role, "both"), role
    # `both` must be a strict superset of `full`: the full instance plus siblings.
    assert set(_declared_fleet_ports("frontdoor", "both")) > set(
        _declared_fleet_ports("frontdoor", "full")
    )


def test_alias_roles_inherit_host_full_fleet_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    """WP-13: alias roles (shared_with_first_n) ride the host's llama-server(s) so
    they must inherit the host's FULL serving fleet, not just the instances they
    were tagged onto (shared_with_first_n_count). Host roles are unchanged."""
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "both")

    _aliases, roles = _stack_manifest_info()

    # SSU-F5: the property under test is a RELATION (alias rides host), so it is
    # asserted as one. It used to be spelled as three pairs of port literals --
    # worker_general [8072, 8082, 8182] as a HOST, coder_escalation -> 8083,
    # worker_summarize -> frontdoor -- which meant each lineup change rewrote the
    # test instead of exercising it. worker_general is an ALIAS since 2026-09-22
    # and the relation still holds, unedited.
    alias_map = _lineup_alias_map()
    assert alias_map, "no alias resolves at all: the lineup source is unreadable"

    # Every host keeps its own declared fleet, and its primary is that fleet's
    # first port.
    for host in sorted(set(alias_map.values())):
        assert roles[host]["ports"] == _declared_fleet_ports(host, "both"), host
        assert roles[host]["port"] == _declared_primary_port(host, "both"), host

    # Every alias inherits its host's WHOLE fleet (previously a single quarter),
    # and its primary port/url resolve to the host's primary.
    for alias, host in sorted(alias_map.items()):
        assert roles[alias]["ports"] == roles[host]["ports"], alias
        assert roles[alias]["port"] == roles[host]["port"], alias
        assert roles[alias]["url"] == roles[host]["url"], alias
        assert _aliases[alias] == host, alias

    # Recompute-and-diff across the repo boundary: the launcher's alias map must
    # agree with the one derived from the registry's own `server_mode.*.shared_with`.
    # Reuses the resolver in scripts/validate/check_shared_with_derivations.py so
    # there is ONE implementation of "which server does this role resolve to".
    from scripts.validate.check_shared_with_derivations import (
        DEFAULT_LAUNCH_MANIFEST,
        DEFAULT_STACK_TOPOLOGY,
        derive as _derive_shared_with,
        load_sources as _load_lineup_sources,
    )

    _lean_registry = (
        Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml"
    )
    _declared_aliases = _derive_shared_with(
        _load_lineup_sources(
            _lean_registry, DEFAULT_LAUNCH_MANIFEST, DEFAULT_STACK_TOPOLOGY
        )
    ).alias_to_host
    for alias, host in alias_map.items():
        # `.get(alias, host)` tolerates a launcher_only_alias (a name the launcher
        # carries with a declared parity exception); a name the registry DOES
        # declare must agree exactly.
        assert _declared_aliases.get(alias, host) == host, alias


def test_serving_record_projects_alias_host_fleet_full_url() -> None:
    """WP-13: _serving_record over an alias launch record emits the full host
    fleet as serving.ports, and stack_prior_serving_url_value emits the
    ``full:``-prefixed fleet URL."""
    host_fleet = [8072, 8082, 8182, 8282, 8382]
    alias_launch_cfg = {
        "tier": "hot",
        "port": host_fleet[0],
        "ports": list(host_fleet),
        "url": "http://localhost:8072",
        "effective_context_tokens": 16384,
        "launch": _launch_record([]),
    }

    serving = _serving_record(
        "worker_math",
        {},
        None,
        None,
        None,
        "stack_manifest.alias->worker_general",
        alias_launch_cfg,
    )

    assert serving["ports"] == host_fleet
    assert stack_prior_serving_url_value(serving) == (
        "full:http://localhost:8072,http://localhost:8082,"
        "http://localhost:8182,http://localhost:8282,http://localhost:8382"
    )


def test_regenerated_worker_math_url_byte_equals_fix_a_delegated_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CRITICAL ACCEPTANCE (WP-13 fleet convergence): a FUTURE regeneration of the
    stack priors yields a worker_math serving URL byte-identical to the operative
    Fix-A delegated field value (models.ServerConfig.worker_math ->
    _server_url_default('worker_general') -> the full worker fleet). Proves the
    generator and the operative layer converge on the same wire value, so a
    deploy of the regenerated artifact is a no-op on worker_math's URL."""
    from src.config.models import _LEGACY_SERVER_URL_FALLBACKS

    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "both")

    _aliases, roles = _stack_manifest_info()
    serving = _serving_record(
        "worker_math",
        {},
        None,
        None,
        None,
        "stack_manifest.alias->worker_general",
        roles["worker_math"],
    )
    regenerated = stack_prior_serving_url_value(serving)

    # SSU-F5: worker_math has no URL of its own -- it DELEGATES to whichever role
    # hosts its process, which is the entire point of "delegated value" in this
    # test's name. Reading `_LEGACY_SERVER_URL_FALLBACKS["worker_math"]` asserted
    # against the alias's own restatement of that delegation; resolving the host
    # first asserts against the delegation itself, and follows a lineup change.
    host = _lineup_host_of("worker_math")
    fix_a_delegated = _LEGACY_SERVER_URL_FALLBACKS[host]
    assert regenerated == fix_a_delegated
    # Not a literal: the generated URL is the host's whole declared fleet, in
    # order, in the `full:` wire form the operative layer emits.
    assert regenerated == "full:" + ",".join(
        f"http://localhost:{port}" for port in _declared_fleet_ports("worker_math", "both")
    )
    # DELIBERATE PIN, do not "modernise": `full:` is the wire prefix the operative
    # layer and the generator must both emit for a multi-instance fleet. A single
    # instance is emitted bare (no prefix), which is why this is asserted on a
    # role whose host declares a fleet.
    assert regenerated.startswith("full:")


def test_alias_without_host_fleet_falls_back_to_own_launch_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP-13: when the primary/host has no resolved port fleet, the alias falls
    back to its own launch ports (prior behavior preserved)."""
    from scripts.server import stack_manifest

    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    monkeypatch.setattr(
        stack_manifest,
        "ROLE_LAUNCH_META",
        {
            "synth_primary": {
                "tier": "hot",
                "mode": "default",
                "no_numa": True,
                "port": None,
                "shared_with_first_n": ["synth_alias"],
            }
        },
    )
    # synth_primary launches no server (no ports resolvable, PORT_MAP empty);
    # synth_alias appears on its own launch server at 9191.
    monkeypatch.setattr(
        stack_manifest, "HOT_SERVERS", [{"port": 9191, "roles": ["synth_alias"]}]
    )
    monkeypatch.setattr(stack_manifest, "WARM_SERVERS", [])
    monkeypatch.setattr(stack_manifest, "PORT_MAP", {})

    _aliases, roles = _stack_manifest_info()

    assert roles["synth_primary"]["ports"] == []
    # Host fleet absent -> alias keeps its own launch port (fallback preserved).
    assert roles["synth_alias"]["ports"] == [9191]
    assert roles["synth_alias"]["port"] == 9191
    assert roles["synth_alias"]["url"] == "http://localhost:9191"
    assert _aliases["synth_alias"] == "synth_primary"


def test_runtime_stack_prior_helpers_fail_closed_on_missing_artifact(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    assert load_stack_priors_artifact(missing) is None
    assert live_stack_role_records(missing) == {}
    assert live_stack_role_ids(missing) == []
    assert live_warm_worker_slots(missing) == {}
    assert live_role_primary_ports(frozenset({"worker_vision"}), missing) == {}
    assert live_stack_lock_role_sets(missing) is None
    assert live_stack_safe_non_stream_roles(missing, min_mem_gb=64.0) is None
    assert live_stack_serving_url_values(missing) == {}
    assert live_stack_serving_slot_limits(missing) == {}


def test_runtime_stack_prior_helpers_project_live_roles(tmp_path: Path) -> None:
    priors = _write_yaml(
        tmp_path / "stack_priors.yaml",
        {
            "roles": {
                "worker_batch": {
                    "deployment_status": "live_stack",
                    "serving": {"tier": "warm", "slots": 4, "ports": [9123]},
                },
                "worker_general": {
                    "deployment_status": "live_stack",
                    "serving": {"tier": "hot", "slots": 4, "ports": [8072]},
                },
                "worker_vision": {
                    "deployment_status": "live_stack",
                    "serving": {"endpoint": "http://127.0.0.1:9101", "ports": [9999]},
                },
                "vision_escalation": {
                    "deployment_status": "live_stack",
                    "serving": {"ports": [9107]},
                },
                "candidate_worker": {
                    "deployment_status": "benchmark_or_candidate",
                    "serving": {"tier": "warm", "slots": 8, "ports": [9000]},
                },
            }
        },
    )

    assert sorted(live_stack_role_records(priors)) == [
        "vision_escalation",
        "worker_batch",
        "worker_general",
        "worker_vision",
    ]
    assert canonical_stack_role_id("worker_explore") == "worker_general"
    assert canonical_stack_role_id(_RETIRED_ARCHITECT_ROLE) == "architect_general"
    assert canonical_stack_role_id("unknown_role") is None
    assert live_stack_role_ids(
        priors,
        preferred_order=["frontdoor", "worker_general", "vision_escalation"],
    ) == [
        "worker_general",
        "vision_escalation",
        "worker_batch",
        "worker_vision",
    ]
    assert live_warm_worker_slots(priors) == {"worker_batch": 4}
    assert live_role_primary_ports(
        frozenset({"worker_vision", "vision_escalation", "candidate_worker"}),
        priors,
    ) == {"worker_vision": 9101, "vision_escalation": 9107}
    assert stack_prior_endpoint_port({"endpoint": "http://localhost:1234/v1"}) == 1234
    assert stack_prior_primary_port(
        {"endpoint": "http://localhost:1234/v1", "ports": [5678]}
    ) == 1234
    assert stack_prior_primary_port({"endpoint": "http://localhost:notaport", "ports": [5678]}) == 5678
    assert stack_prior_primary_port({"endpoint": "http://localhost:notaport"}) is None
    assert stack_prior_serving_ports({"ports": [1, "2", 3, None]}) == [1, 3]
    assert stack_prior_serving_url_value({"ports": [9100, 9200]}) == (
        "full:http://localhost:9100,http://localhost:9200"
    )
    assert stack_prior_serving_url_value({"endpoint": "http://localhost:9300"}) == (
        "http://localhost:9300"
    )
    assert live_stack_serving_url_values(priors) == {
        "worker_batch": "http://localhost:9123",
        "worker_general": "http://localhost:8072",
        "worker_vision": "http://localhost:9999",
        "vision_escalation": "http://localhost:9107",
    }
    assert live_stack_serving_slot_limits(priors) == {
        "http://localhost:9123": 4,
        "http://localhost:8072": 4,
    }


def test_live_stack_slot_query_ports_filters_non_llama_and_aliases(tmp_path: Path) -> None:
    priors = _write_yaml(
        tmp_path / "stack_priors.yaml",
        {
            "roles": {
                "frontdoor": {
                    "deployment_status": "live_stack",
                    "serving": {
                        "binary": "llama.cpp",
                        "launch": {
                            "entries": [
                                {"port": 8070, "alias": False},
                                {"port": 8080, "alias": False},
                            ]
                        },
                    },
                },
                "coder_escalation": {
                    "deployment_status": "live_stack",
                    "serving": {
                        "binary": "llama.cpp",
                        "launch": {"entries": [{"port": 8070, "alias": True}]},
                    },
                },
                "worker_general": {
                    "deployment_status": "live_stack",
                    "serving": {
                        "binary": "ik-pr1744",
                        "launch": {
                            "runtime": {
                                "binary_path": "/mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server"
                            },
                            "entries": [{"port": 8072, "alias": False}],
                        },
                    },
                },
                "reap_candidate": {
                    "deployment_status": "benchmark_only",
                    "serving": {
                        "binary": "llama.cpp",
                        "launch": {"entries": [{"port": 8099, "alias": False}]},
                    },
                },
                "embedder": {
                    "deployment_status": "live_stack",
                    "serving": {
                        "binary": "embedding-server",
                        "launch": {"entries": [{"port": 8090, "alias": False}]},
                    },
                },
            }
        },
    )

    assert live_stack_slot_query_ports(priors) == {
        "frontdoor": [8070, 8080],
        "worker_general": [8072],
    }


def test_runtime_stack_prior_policy_helpers_project_launch_and_memory(tmp_path: Path) -> None:
    priors = _write_yaml(
        tmp_path / "stack_priors.yaml",
        {
            "roles": {
                "frontdoor": {
                    "deployment_status": "live_stack",
                    "serving": {"launch": {"modes": ["default"], "entries": []}},
                    "model": {"mem_gb": 37.0},
                },
                "worker_general": {
                    "deployment_status": "live_stack",
                    "serving": {"launch": {"modes": ["worker_pool"], "entries": []}},
                    "model": {"mem_gb": 16.0},
                },
                "worker_vision": {
                    "deployment_status": "live_stack",
                    "serving": {"launch": {"entries": [{"vision_type": "worker"}]}},
                    "model": {"mem_gb": 22.0},
                },
                "architect_general": {
                    "deployment_status": "live_stack",
                    "serving": {"launch": {"modes": ["default"], "entries": []}},
                    "model": {"mem_gb": 69.0},
                },
                "candidate_large": {
                    "deployment_status": "benchmark_or_candidate",
                    "serving": {"launch": {"modes": ["worker_pool"], "entries": []}},
                    "model": {"mem_gb": 120.0},
                },
            }
        },
    )
    records = live_stack_role_records(priors)

    assert stack_prior_launch_modes(records["worker_general"]) == {"worker_pool"}
    assert stack_prior_launch_entries(records["worker_vision"]) == [{"vision_type": "worker"}]
    assert stack_prior_uses_shared_worker_launch(records["worker_general"]) is True
    assert stack_prior_uses_shared_worker_launch(records["worker_vision"]) is True
    assert stack_prior_uses_shared_worker_launch(records["frontdoor"]) is False
    assert stack_prior_model_mem_gb(records["architect_general"]) == 69.0

    lock_roles = live_stack_lock_role_sets(priors)
    assert lock_roles is not None
    heavy, light = lock_roles
    assert {"frontdoor", "architect_general"} <= heavy
    assert {"worker_general", "worker_vision"} <= light
    assert "candidate_large" not in heavy
    assert "candidate_large" not in light
    assert live_stack_safe_non_stream_roles(priors, min_mem_gb=64.0) == frozenset(
        {"architect_general"}
    )


def test_compile_prefers_server_mode_for_shared_role_memory_and_serving(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "registry.yaml",
        {
            "server_mode": {
                "frontdoor": {
                    "url": "http://localhost:8070",
                    "port": 8070,
                    "tier": "hot",
                    "throughput": 24.3,
                },
                "coder_escalation": {
                    "url": "http://localhost:8070",
                    "port": 8070,
                    "tier": "hot",
                    "throughput": 24.3,
                },
            },
            "roles": {
                "frontdoor": {"memory": {"residency": "warm"}},
                "coder_escalation": {"memory": {"residency": "warm"}},
            },
        },
    )
    descriptor_path = _write_yaml(
        tmp_path / "descriptors.yaml",
        {
            "models": [
                {
                    "model_id": "qwen3.6-35b-a3b-q8",
                    "display_name": "Qwen3.6 Q8",
                    "family": "qwen3.6",
                    "arch": "moe",
                    "params_b": 35,
                    "active_b": 3,
                    "quant": "Q8_0",
                    "mem_gb": 37,
                    "ctx_max": 131072,
                    "architecture": {"n_layers": 64, "attention_layers": 16},
                    "modalities": ["text"],
                    "role_bindings": {
                        "roles": ["frontdoor", "coder_escalation"],
                        "server_roles": ["frontdoor", "coder_escalation"],
                        "shared_mmap": True,
                    },
                    "quality": {
                        "suite_vector": {"overall": 0.929},
                        "measured": [{"date": "2026-05-04"}],
                    },
                    "speed": {"solo_96t_tps": 24.3, "measured": [{"value_tps": 24.3}]},
                    "acceleration": {"spec_type": "none"},
                    "serving": {"binary": "llama.cpp", "ports": [8070]},
                    "known_gaps": [],
                }
            ]
        },
    )

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"frontdoor", "coder_escalation"},
    )

    frontdoor = priors["roles"]["frontdoor"]
    coder = priors["roles"]["coder_escalation"]
    assert priors["status"] == "compiled"
    assert priors["stack_priors_version"] == STACK_PRIORS_VERSION
    assert priors["contract"]["schema"] == "epyc.stack_priors"
    # Derived, not restated. A literal pin list here is a SECOND copy of the
    # compiler's own `source_artifacts` block, and that duplication has already
    # failed once: scripts/validate/stack_change_guard.py carried the same
    # 7-item list while the compiler emitted 9, so launch_manifest.yaml and
    # stack_topology.yaml — the files that now hold the launcher configuration
    # the .py loaders used to contain — were pinned and never verified
    # (a517793c fixed the guard by iterating the producer's keys and demoting
    # the list to a FLOOR). This test is the last surviving restatement.
    from scripts.validate.stack_change_guard import REQUIRED_SOURCE_ARTIFACTS

    source_artifacts = priors["source_artifacts"]
    # (1) The floor the guard enforces: a pin the compiler silently stopped
    #     emitting is still caught.
    assert set(source_artifacts) >= set(REQUIRED_SOURCE_ARTIFACTS), (
        f"compiler dropped a required pin: "
        f"{sorted(set(REQUIRED_SOURCE_ARTIFACTS) - set(source_artifacts))}"
    )
    # (2) The property actually worth guarding: every pin is a complete
    #     provenance triple, so the chain can be verified rather than trusted.
    for label, pin in source_artifacts.items():
        assert set(pin) == {"path", "sha256", "repo_commit"}, label
        assert pin["path"] and pin["sha256"], label
    # (3) Cross-check against the SHIPPED compiled artifact, which is this
    #     compiler's own committed output: a pin added or removed without a
    #     recompile shows up as a set mismatch here.
    shipped = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "orchestration"
            / "derived"
            / "stack_priors.yaml"
        ).read_text(encoding="utf-8")
    )
    assert set(source_artifacts) == set(shipped["source_artifacts"])
    assert validate_stack_priors_contract(priors) == []
    assert frontdoor["priors"]["memory_cost"] == 1.0
    assert frontdoor["evidence"]["precedence"]["memory_cost"] == "server_mode.tier"
    assert frontdoor["model"]["n_layers"] == 64
    assert frontdoor["model"]["attention_layers"] == 16
    frontdoor_runtime = frontdoor["serving"]["launch"]["runtime"]
    assert frontdoor_runtime["binary_family"] == "llama.cpp"
    # This synthetic registry declares no `slots`, so the compiler falls back to
    # the DECLARED default in orchestration/launch_manifest.yaml
    # (launch_shape.fallback_slots.default) — read here rather than restated.
    # The literal `1` that stood here was the SERIAL_ROLES clamp, deliberately
    # removed on 2026-08-02: it applied an ADMISSION policy to a SERVING number,
    # so this record shipped two disagreeing slot counts (serving.slots vs
    # runtime.cache.slots, the latter becoming the launcher's `-np`). Serial
    # admission now lives in src/api/admission.py; frontdoor is still in
    # launch_manifest serial_roles, which is exactly why re-pinning 1 here would
    # resurrect the clamp in test form.
    _launch_manifest = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "orchestration"
            / "launch_manifest.yaml"
        ).read_text(encoding="utf-8")
    )
    assert (
        frontdoor_runtime["cache"]["slots"]
        == _launch_manifest["launch_shape"]["fallback_slots"]["default"]
    )
    # K4 (handoffs/active/dynamic-stack-concurrency.md): read the DECLARED
    # default from launch_manifest.yaml rather than restating it as a literal,
    # same rationale as the `slots` assertion above it -- a literal here is
    # exactly what let 8192 (clamped inert to 2048 by llama.cpp's n_batch
    # default with no -b emitted) go unnoticed.
    assert (
        frontdoor_runtime["cache"]["ubatch"]
        == _launch_manifest["launch_shape"]["default_ubatch_tokens"]
    )
    assert frontdoor_runtime["cache"]["kv_type_k"] == "q8_0"
    assert frontdoor_runtime["cache"]["kv_type_v"] == "q8_0"
    assert frontdoor_runtime["cache"]["mlock"] is True
    assert frontdoor_runtime["cache"]["slot_save_path"].endswith("/kv_slots/frontdoor")
    assert frontdoor_runtime["flags"]["jinja"] is True
    assert frontdoor_runtime["flags"]["spec"]["enabled"] is False
    # 2026-08-01 W1 cutover: was http://localhost:8070. The endpoint is resolved
    # from the REAL launcher manifest (stack_manifest.PORT_MAP), not from this
    # synthetic registry — which is exactly the drift this assertion detects.
    # coder_escalation is now an alias on architect_general's :8083 process.
    assert coder["serving"]["endpoint"] == "http://localhost:8083"
    assert coder["serving"]["shared_mmap"] is True


def test_compile_maps_model_role_server_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``server_mode.<server>.model_role`` binds a server row to a role.

    SSU-F5: this test used to describe a synthetic ``worker`` server in its
    registry and then assert a RUNTIME RECORD that the compiler resolves from the
    REAL launcher -- so ten of its assertions (ports, primary role, mode,
    requirements, slots, ubatch, no_mmap, spec) were restatements of the live
    lineup, and every one of them went stale on 2026-09-22 when worker_general
    stopped hosting a server. The fixture now DECLARES the launcher lineup it
    needs (a self-hosted worker-pool worker_general, which is what the registry
    half always described), so the record under test is a function of the
    fixture. The live-lineup path is covered by the tests above, which derive
    their expectations instead of pinning them.
    """
    from scripts.server import stack_manifest

    worker_pool_model = "/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf"
    explore_draft_model = "/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf"
    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    monkeypatch.setattr(
        stack_manifest,
        "ROLE_LAUNCH_META",
        {
            "worker_general": {
                "tier": "hot",
                "mode": "worker_pool",
                "worker_type": "explore",
            }
        },
    )
    monkeypatch.setattr(
        stack_manifest,
        "HOT_SERVERS",
        [
            {
                "port": 8072,
                "roles": ["worker_general"],
                "numa_instance": 0,
                # the launcher's server dicts flag the mode as a boolean key
                # (`worker_pool` / `vision` / `embedding`); `_launch_mode_for_server`
                # reads exactly these.
                "worker_pool": True,
                "worker_type": "explore",
            }
        ],
    )
    monkeypatch.setattr(stack_manifest, "WARM_SERVERS", [])
    monkeypatch.setattr(stack_manifest, "PORT_MAP", {"worker_general": 8072})
    monkeypatch.setattr(
        stack_manifest, "WORKER_POOL_MODELS", {"explore": worker_pool_model}
    )
    monkeypatch.setattr(stack_manifest, "EXPLORE_DRAFT_MODEL", explore_draft_model)

    registry_path = _write_yaml(
        tmp_path / "registry.yaml",
        {
            "server_mode": {
                "worker": {
                    "url": "http://localhost:8072",
                    "port": 8072,
                    "tier": "hot",
                    "model_role": "worker_general",
                    "throughput": "60.7",
                }
            },
            "roles": {"worker_general": {"memory": {"residency": "warm"}}},
        },
    )
    descriptor_path = _write_yaml(
        tmp_path / "descriptors.yaml",
        {
            "models": [
                {
                    "model_id": "gemma4-26b-a4b-q4",
                    "role_bindings": {"roles": ["worker_general"], "server_roles": ["worker"]},
                    "quality": {"suite_vector": {"overall": 0.9}, "measured": []},
                    "speed": {"quarter_48t_tps": 60.7, "measured": []},
                    "acceleration": {"spec_type": "mtp"},
                    "serving": {"ports": [8072], "binary": "llama.cpp"},
                    "known_gaps": [],
                }
            ]
        },
    )

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_general"},
    )

    worker = priors["roles"]["worker_general"]
    assert worker["serving"]["server_role"] == "worker"
    assert worker["serving"]["binding"] == "server_mode.model_role"
    # SSU-F5: DERIVED, not restated. This synthetic registry declares port 8072,
    # but the endpoint is resolved from the REAL launcher manifest -- which is the
    # binding this test is about. The literal 8072 that stood here was a copy of
    # `port_map` and went red when the 2026-09-22 lineup change moved
    # worker_general onto frontdoor's process, proving nothing about the binding.
    assert worker["serving"]["ports"] == _declared_fleet_ports("worker_general")
    # 16384 -> 262144 (2026-08-02, operator-ratified). DERIVED from
    # server_mode.worker.serving_shape.n_ctx, which reaches worker_general through
    # the model_role binding this test is about. The 16384 it used to read was the
    # stale `roles.worker_general.model.max_context`; the gemma4 GGUF's own
    # context_length is 262144.
    assert worker["serving"]["effective_context_tokens"] == 262144
    # SSU-F5: the primary role and the launch mode are the SAME lineup fact as
    # the port above -- which process this role rides -- so they are derived from
    # the same place. The pair ["worker_general"] / ["worker_pool"] described the
    # pre-2026-09-22 lineup, when worker_general ran its own worker-pool server.
    assert worker["serving"]["launch"]["primary_roles"] == [
        _lineup_host_of("worker_general")
    ]
    assert worker["serving"]["launch"]["modes"] == [_lineup_launch_mode("worker_general")]
    # A worker-pool launch takes its paths from the launcher's declared pool
    # model + explore drafter, so they are read from the declaration rather than
    # spelled out. The gemma-4 literals that stood here restated
    # stack_manifest.WORKER_POOL_MODELS, which has since changed model twice.
    assert worker["serving"]["launch"]["requirements"] == {
        "model_path": worker_pool_model,
        "draft_model_path": explore_draft_model,
    }
    runtime = worker["serving"]["launch"]["runtime"]
    assert runtime["binary_family"] == "llama.cpp"
    # 2026-09-21: assert the binary is whatever the KERNEL STORE resolves, not a
    # specific build tree. This previously pinned "/llama.cpp/build/bin/llama-server",
    # which encoded the pre-v10 state where production/cpu pointed into the frozen
    # source tree. A promotion repoints that symlink by design, so a tree-path literal
    # here would fail on every future promotion while proving nothing about the
    # binding under test (which is that the role's device selects the backend).
    from src.registry.kernel_paths import server_binary as _store_server_binary

    assert runtime["binary_path"] == str(_store_server_binary("cpu"))
    assert runtime["binary_path"].endswith("/llama-server")
    # 2026-07-31: binary_dir is now ALWAYS resolved, from the role's declared device
    # via the stable kernel layer, instead of being left None and letting a CPU-only
    # literal decide. A CPU role resolves to the cpu backend; a role declaring
    # device: ROCm0 resolves to the gpu backend, which is what makes that
    # declaration reach the launcher at all.
    assert runtime["binary_dir"] is not None
    from src.registry.kernel_paths import backend_dir as _store_backend_dir

    assert runtime["binary_dir"] == str(_store_backend_dir("cpu"))
    assert runtime["ld_library_path"] == []
    # A DERIVED backend must not carry the env consequences of an EXPLICIT
    # registry binary_dir override — otherwise every role silently changes policy.
    assert runtime["env_policy"] == "canonical"
    assert runtime["kmp_blocktime"] is None
    # Same 16384 -> 262144 move as serving.effective_context_tokens above; this is
    # the runtime half of the same number, and the two must not diverge.
    assert runtime["cache"]["context_tokens"] == 262144
    # Slots come from the launcher's DECLARED per-mode fallback for this launch
    # shape, not from a pinned 1 (launch_shape.fallback_slots.worker_pool).
    assert (
        runtime["cache"]["slots"]
        == stack_manifest.fallback_slots_for_mode(
            mode="worker_pool", worker_type="explore"
        ).slots
    )
    # DELIBERATE PIN: 512 is the canonical explore-recipe micro-batch, the
    # compiler's own `fallback=512` for a worker_pool+explore launch with no
    # declared ubatch. It is a property of the RECIPE, not of the lineup, so it
    # does not follow a topology change and must not be "derived" from one.
    assert runtime["cache"]["ubatch"] == 512
    # KV quant is declared per role by the launcher; read it rather than restate.
    assert (
        runtime["cache"]["kv_type_k"],
        runtime["cache"]["kv_type_v"],
    ) == stack_manifest.LAUNCH_KV_QUANT_CONFIGS["worker_general"]
    # no_mmap True is the worker_pool+explore canonical-recipe default (fixture),
    # mlock follows the topology's declared MLOCK_ROLES for this role.
    assert runtime["cache"]["no_mmap"] is True
    from scripts.server.stack_numa import MLOCK_ROLES

    assert runtime["cache"]["mlock"] is ("worker_general" in MLOCK_ROLES)
    assert runtime["flags"]["jinja"] is True
    assert runtime["flags"]["reasoning"] == "off"
    assert runtime["flags"]["spec"]["enabled"] is True
    assert runtime["flags"]["spec"]["type"] == "draft-mtp"
    assert runtime["flags"]["spec"]["draft_max"] == 2
    assert runtime["flags"]["spec"]["draft_p_min"] == 0.0
    assert runtime["flags"]["spec"]["threads_draft"] == 16
    assert worker["priors"]["throughput_tps"] == 60.7
    assert worker["priors"]["memory_cost"] == 1.0


def test_compile_prefers_server_mode_launch_requirement_paths(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "registry.yaml",
        {
            "server_mode": {
                "worker": {
                    "url": "http://localhost:8072",
                    "port": 8072,
                    "tier": "hot",
                    "model_role": "worker_general",
                    "model_path": "/models/gemma-4-26B-A4B-it-Q8_0.gguf",
                    "draft_model_path": "/models/gemma-4-26B-A4B-it-draft-Q8_0.gguf",
                }
            },
            "roles": {"worker_general": {"memory": {"residency": "warm"}}},
        },
    )
    descriptor_path = _write_yaml(
        tmp_path / "descriptors.yaml",
        {
            "models": [
                {
                    "model_id": "gemma4-26b-a4b-q8",
                    "role_bindings": {"roles": ["worker_general"], "server_roles": ["worker"]},
                    "quality": {"suite_vector": {"overall": 0.9}, "measured": []},
                    "speed": {"quarter_48t_tps": 60.7, "measured": []},
                    "acceleration": {"spec_type": "mtp"},
                    "serving": {"ports": [8072], "binary": "ik-pr1744"},
                    "known_gaps": [],
                }
            ]
        },
    )

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_general"},
    )

    launch = priors["roles"]["worker_general"]["serving"]["launch"]
    assert launch["requirements"]["model_path"] == "/models/gemma-4-26B-A4B-it-Q8_0.gguf"
    assert (
        launch["requirements"]["draft_model_path"]
        == "/models/gemma-4-26B-A4B-it-draft-Q8_0.gguf"
    )
    # SSU-F5: the runtime spec block is a function of the LINEUP, not of this
    # fixture. A role that launches its own process carries the drafter declared
    # in its requirements; an ALIAS rides its host's process and carries no
    # drafter of its own (worker_general became an alias on 2026-09-22, which is
    # what turned this literal into a failure). Both branches are asserted, so the
    # test keeps its meaning whichever side of the lineup worker_general is on.
    spec = launch["runtime"]["flags"]["spec"]
    if _lineup_is_alias("worker_general"):
        assert spec["enabled"] is False
        assert spec["draft_model_path"] is None
    else:
        assert spec["draft_model_path"] == launch["requirements"]["draft_model_path"]


def test_compile_shared_aliases_use_runtime_descriptor(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "registry.yaml",
        {
            "server_mode": {
                "worker": {
                    "url": "http://localhost:8072",
                    "port": 8072,
                    "tier": "hot",
                    "slots": 1,
                    "model_role": "worker_general",
                    "shared_with": ["worker_math", "toolrunner"],
                    "throughput": 60.7,
                    "numa_ports": [8082],
                }
            },
            "roles": {
                "worker_general": {"memory": {"residency": "hot"}},
                "worker_math": {"memory": {"residency": "hot"}},
                "toolrunner": {"memory": {"residency": "hot"}},
            },
        },
    )
    descriptor_path = _write_yaml(
        tmp_path / "descriptors.yaml",
        {
            "models": [
                {
                    "model_id": "gemma4-26b-a4b-q4",
                    "role_bindings": {
                        "roles": ["worker_general", "worker_math", "toolrunner"],
                        "server_roles": ["worker"],
                        "shared_mmap": True,
                        "alias_overrides": [
                            {
                                "role": "worker_math",
                                "served_by": "worker_general",
                                "ignored_model_id": "qwen2.5-math-7b-q4_k_m",
                                "reason": "server_mode.shared_with runtime takes precedence",
                            }
                        ],
                    },
                    "quality": {"suite_vector": {"overall": 0.9}, "measured": []},
                    "speed": {"quarter_48t_tps": 60.7, "measured": []},
                    "acceleration": {
                        "spec_type": "mtp",
                        "draft_compat": ["gemma4-26b-a4b-assistant-q8"],
                    },
                    "serving": {"ports": [8072, 8082], "binary": "ik-pr1744"},
                    "known_gaps": [],
                }
            ]
        },
    )

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_general", "worker_math", "toolrunner"},
        allow_incomplete=True,
    )

    for role in ("worker_general", "worker_math", "toolrunner"):
        record = priors["roles"][role]
        assert record["model_id"] == "gemma4-26b-a4b-q4"
        assert record["acceleration"]["spec_type"] == "mtp"
        assert record["priors"]["throughput_tps"] == 60.7
        assert not any(gap.startswith("Role-server conflict:") for gap in record["known_gaps"])
        assert record["known_gaps"] == []
        assert record["evidence"]["alias_overrides"] == [
            {
                "role": "worker_math",
                "served_by": "worker_general",
                "ignored_model_id": "qwen2.5-math-7b-q4_k_m",
                "reason": "server_mode.shared_with runtime takes precedence",
            }
        ]

    # SSU-F5: DERIVED from the real launcher declaration (this synthetic registry
    # says 8072; the compiler resolves ports from the launcher). The three roles
    # must land on the SAME fleet -- that co-residency is what the test asserts --
    # and that fleet is whatever the lineup currently declares.
    for role in ("worker_general", "worker_math", "toolrunner"):
        assert priors["roles"][role]["serving"]["ports"] == _declared_fleet_ports(role), role
    assert (
        priors["roles"]["worker_math"]["serving"]["ports"]
        == priors["roles"]["worker_general"]["serving"]["ports"]
        == priors["roles"]["toolrunner"]["serving"]["ports"]
    )
    # 16384 -> 262144 (2026-08-02): both aliases ride worker's process and inherit
    # its serving_shape.n_ctx, which is the property this test asserts.
    assert priors["roles"]["worker_math"]["serving"]["effective_context_tokens"] == 262144
    assert priors["roles"]["toolrunner"]["serving"]["effective_context_tokens"] == 262144
    assert priors["roles"]["worker_math"]["serving"]["binding"] == "server_mode.shared_with"
    assert priors["roles"]["toolrunner"]["serving"]["binding"] == "server_mode.shared_with"
    assert priors["roles"]["worker_math"]["serving"]["launch"]["requirements"] == (
        priors["roles"]["worker_general"]["serving"]["launch"]["requirements"]
    )
    assert priors["roles"]["worker_math"]["serving"]["launch"]["runtime"] == (
        priors["roles"]["worker_general"]["serving"]["launch"]["runtime"]
    )


def test_launch_runtime_record_canonicalizes_worker_explore_kv_types() -> None:
    runtime = _launch_runtime_record(
        role="worker_explore",
        descriptor={},
        server_cfg=None,
        role_cfg=None,
        launch_cfg={
            "launch": {
                "primary_roles": ["worker_explore"],
                "modes": ["worker_pool"],
                "requirements": {"model_path": "/models/gemma.gguf"},
                "runtime": {},
            }
        },
    )

    assert runtime["cache"]["kv_type_k"] == "q8_0"
    assert runtime["cache"]["kv_type_v"] == "q8_0"


def test_launch_runtime_record_derives_reasoning_off_from_thinking_prior() -> None:
    runtime = _launch_runtime_record(
        role="frontdoor",
        descriptor={
            "acceleration": {
                "enable_thinking": False,
                "thinking_control": {
                    "mode": "toggle_off",
                    "source": "model.disable_thinking",
                },
            },
        },
        server_cfg={},
        role_cfg={},
        launch_cfg={
            "effective_context_tokens": 8192,
            "launch": {
                "primary_roles": ["frontdoor"],
                "modes": ["default"],
                "requirements": {},
                "runtime": {},
            },
        },
    )

    assert runtime["flags"]["reasoning"] == "off"


def test_launch_runtime_record_does_not_force_reasoning_when_template_ignores_toggle() -> None:
    runtime = _launch_runtime_record(
        role="ingest_long_context",
        descriptor={
            "acceleration": {
                "enable_thinking": None,
                "thinking_control": {
                    "mode": "template_ignores_enable_thinking",
                    "source": "registry note",
                },
            },
        },
        server_cfg={},
        role_cfg={},
        launch_cfg={
            "effective_context_tokens": 131072,
            "launch": {
                "primary_roles": ["ingest_long_context"],
                "modes": ["default"],
                "requirements": {},
                "runtime": {},
            },
        },
    )

    assert runtime["flags"]["reasoning"] is None
    # 1 -> 2. This fixture passes an empty server_cfg, so `slots` comes from the
    # manifest's declared `launch_shape.fallback_slots.default` (2). It read 1
    # only because the SERIAL_ROLES clamp — an ADMISSION policy applied to a
    # SERVING number — used to rewrite it here; that clamp was removed on
    # 2026-08-02 when the operator ratified explicit per-instance slot counts.
    assert runtime["cache"]["slots"] == 2


def test_launch_runtime_record_projects_ap3b_spec_numeric_controls() -> None:
    runtime = _launch_runtime_record(
        role="worker_general",
        descriptor={},
        server_cfg={
            "acceleration": {
                "type": "speculative_decoding",
                "spec_type": "ngram-mod,draft-mtp",
                "draft_max": 5,
                "draft_min": 0,
                "draft_p_min": 0.125,
                "draft_p_split": 0.5,
                "threads_draft": 12,
                "ngram_mod_n_min": 0,
                "ngram_mod_n_max": 96,
                "ngram_mod_n_match": 16,
            }
        },
        role_cfg=None,
        launch_cfg={
            "launch": {
                "primary_roles": ["worker_general"],
                "modes": ["worker_pool"],
                "entries": [{"worker_type": "explore"}],
                "requirements": {
                    "model_path": "/models/gemma.gguf",
                    "draft_model_path": "/models/draft.gguf",
                },
                "runtime": {},
            }
        },
    )

    spec = runtime["flags"]["spec"]
    assert spec["enabled"] is True
    assert spec["type"] == "ngram-mod,draft-mtp"
    assert spec["draft_model_path"] == "/models/draft.gguf"
    assert spec["draft_max"] == 5
    assert spec["draft_min"] == 0
    assert spec["draft_p_min"] == 0.125
    assert spec["draft_p_split"] == 0.5
    assert spec["threads_draft"] == 12
    assert spec["ngram_mod_n_min"] == 0
    assert spec["ngram_mod_n_max"] == 96
    assert spec["ngram_mod_n_match"] == 16


def test_launch_runtime_record_does_not_inject_vision_escalation_override() -> None:
    launch_cfg = {
        "launch": {
            "primary_roles": ["vision_escalation"],
            "modes": ["vision"],
            "entries": [{"vision_type": "escalation"}],
            "requirements": {
                "model_path": "/models/qwen2.5-vl.gguf",
                "mmproj_path": "/models/qwen2.5-vl-mmproj.gguf",
            },
            "runtime": {},
        }
    }

    runtime = _launch_runtime_record(
        role="vision_escalation",
        descriptor={},
        server_cfg=None,
        role_cfg={
            "server": {"device": "ROCm0", "reasoning": "off"},
            "acceleration": {"type": "baseline"},
        },
        launch_cfg=launch_cfg,
    )

    assert runtime["flags"]["device"] == "ROCm0"
    assert runtime["flags"]["reasoning"] == "off"
    assert runtime["flags"]["override_kv"] == []

    runtime = _launch_runtime_record(
        role="vision_escalation",
        descriptor={},
        server_cfg=None,
        role_cfg={
            "acceleration": {
                "type": "moe_expert_reduction",
                "override_key": "qwen3vlmoe.expert_used_count",
                "experts": 4,
            }
        },
        launch_cfg=launch_cfg,
    )

    assert runtime["flags"]["override_kv"] == ["qwen3vlmoe.expert_used_count=int:4"]


def test_launch_runtime_record_accepts_role_level_runtime_requirements() -> None:
    runtime = _launch_runtime_record(
        role="vision_escalation",
        descriptor={},
        server_cfg=None,
        role_cfg={
            "server": {
                "device": "ROCm0",
                "reasoning": "off",
                "runtime_requirements": {
                    "binary_dir": "/tmp/v7-hip/bin",
                    "ld_library_path": ["/tmp/v7-hip/bin"],
                },
            },
            "acceleration": {"type": "baseline"},
        },
        launch_cfg={
            "launch": {
                "primary_roles": ["vision_escalation"],
                "modes": ["vision"],
                "entries": [{"vision_type": "escalation"}],
                "requirements": {
                    "model_path": "/models/minicpm.gguf",
                    "mmproj_path": "/models/minicpm-mmproj.gguf",
                },
                "runtime": {},
            }
        },
    )

    assert runtime["binary_dir"] == "/tmp/v7-hip/bin"
    assert runtime["binary_path"] == "/tmp/v7-hip/bin/llama-server"
    assert runtime["ld_library_path"] == ["/tmp/v7-hip/bin"]
    assert runtime["env_policy"] == "binary_override_strip_ggml"


def test_server_mode_requirement_overrides_keep_shared_alias_on_served_model() -> None:
    requirements = _server_mode_launch_requirement_overrides(
        "worker_math",
        {
            "model_role": "worker_general",
            "model": "gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf",
            "draft_model": "gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf",
        },
        {
            "model": {
                "path": (
                    "lmstudio-community/Qwen2.5-Math-7B-Instruct-GGUF/"
                    "Qwen2.5-Math-7B-Instruct-Q4_K_M.gguf"
                )
            }
        },
    )

    assert requirements["model_path"] == (
        "/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf"
    )
    assert requirements["draft_model_path"] == (
        "/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf"
    )


def test_compile_preserves_conflicts_as_gaps_when_allowed(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "registry.yaml",
        {"server_mode": {}, "roles": {"worker_math": {"memory": {"residency": "warm"}}}},
    )
    descriptor_path = _write_yaml(
        tmp_path / "descriptors.yaml",
        {
            "models": [
                {
                    "model_id": "qwen2.5-math-7b-q4",
                    "role_bindings": {"roles": ["worker_math"], "server_roles": []},
                    "quality": {"suite_vector": {}, "measured": []},
                    "speed": {"measured": []},
                    "acceleration": {},
                    "serving": {"ports": []},
                    "known_gaps": ["Role-server conflict: stale worker server binding"],
                }
            ]
        },
    )

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_math"},
        allow_incomplete=True,
    )

    role = priors["roles"]["worker_math"]
    assert priors["status"] == "compiled_with_gaps"
    assert role["status"] == "compiled_with_gaps"
    assert "Role-server conflict: stale worker server binding" in role["known_gaps"]
    assert role["serving"]["binding"] == "stack_manifest.alias->stack_manifest.role"
    # SSU-F5: this registry declares NO server_mode at all, so both the port and
    # the primary role fall back to the real launcher -- i.e. they are functions
    # of the lineup and must be read from it. The pair of literals that stood here
    # (8072 / "worker_general") both went stale on 2026-09-22.
    assert role["serving"]["ports"] == _declared_fleet_ports("worker_math")
    assert role["serving"]["launch"]["entries"][0]["alias"] is True
    assert role["serving"]["launch"]["entries"][0]["primary_role"] == _lineup_host_of(
        "worker_math"
    )


def test_compile_uses_stack_manifest_when_server_mode_is_absent(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "registry.yaml",
        {"server_mode": {}, "roles": {"worker_vision": {"memory": {"residency": "warm"}}}},
    )
    descriptor_path = _write_yaml(
        tmp_path / "descriptors.yaml",
        {
            "models": [
                {
                    "model_id": "qwen2.5-vl-7b-q4",
                    "role_bindings": {"roles": ["worker_vision"], "server_roles": ["worker_vision"]},
                    "quality": {"suite_vector": {"overall": 0.81}, "measured": []},
                    "speed": {"solo_96t_tps": 20.0, "measured": []},
                    "acceleration": {},
                    "serving": {"ports": [8086]},
                    "known_gaps": [],
                }
            ]
        },
    )

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_vision"},
    )

    role = priors["roles"]["worker_vision"]
    assert role["deployment_status"] == "live_stack"
    assert role["serving"]["binding"] == "stack_manifest.role"
    assert role["serving"]["endpoint"] == "http://localhost:8086"
    # 2026-08-02 phase 2: was 2, from the `mode == "vision"` literal
    # `1 if vision_type == "escalation" else 2`. That literal is the exhibit for
    # this refactor — it shadowed `server_mode.worker_vision.slots: 1` and made
    # the launcher emit `-np 2` for a role declaring 1. With `server_mode` empty,
    # as here, the value now comes from the DECLARED
    # `launch_shape.fallback_slots.vision` (1) instead of a literal in the
    # compiler. A registry that declares slots still wins over it — that is the
    # `_runtime_flag_int_prior` path, covered by the server_mode tests above.
    assert role["serving"]["slots"] == 1
    # 8192 -> 16384 (2026-08-01 W1) -> 65536 (2026-08-02). LAUNCH_CONTEXT_TOKENS is
    # now DERIVED from server_mode.worker_vision.serving_shape.n_ctx rather than
    # declared launcher-side, so this reads the ratified value even though this
    # fixture's own `server_mode` is empty.
    assert role["serving"]["effective_context_tokens"] == 65536
    # `cpu_shape_class` and per-entry `slots` are new (2026-08-02): the entry is
    # now self-describing — where it runs, on what shape, with how many slots —
    # which is what lets one role's full and halves carry different `-np`.
    # worker_vision has a single GPU_HOST_LANE instance, so its class is that.
    # `slots` is 1 here because this fixture declares no registry slots at all and
    # falls through to launch_shape.fallback_slots.vision, as asserted above.
    assert role["serving"]["launch"]["entries"] == [
        {
            "port": 8086,
            "primary_role": "worker_vision",
            "mode": "vision",
            "alias": False,
            # STACKCHG-DFLASH2-20261003: cold CPU on NUMA_HALF_A (was gpu_host_lane).
            "cpu_shape_class": "half",
            "vision_type": "worker",
            "slots": 1,
        }
    ]
    # 2026-08-01 W1 cutover: was Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf +
    # mmproj-model-f16.gguf; the VL lane moved to Qwen3-VL-30B-A3B on MI210.
    assert role["serving"]["launch"]["requirements"]["model_path"].endswith(
        "Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"
    )
    assert role["serving"]["launch"]["requirements"]["mmproj_path"].endswith(
        "mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf"
    )
    runtime = role["serving"]["launch"]["runtime"]
    assert runtime["binary_family"] == "llama.cpp"
    # Same phase-2 move as serving.slots above: 2 was the `mode == "vision"`
    # literal, 1 is launch_shape.fallback_slots.vision.
    assert runtime["cache"]["slots"] == 1
    assert runtime["cache"]["ubatch"] is None
    assert runtime["cache"]["mlock"] is False
    # Still None, and deliberately so: `device` is a server_mode field and this
    # test's registry has an EMPTY server_mode. The live compile does emit ROCm0
    # for worker_vision after the 2026-08-01 W1 cutover (from
    # server_mode.worker_vision.device); that path is witnessed by
    # tests/unit/test_build_server_command_helpers.py, not here. Asserting ROCm0
    # here would assert the launcher can invent a device the registry never
    # declared, which is the opposite of the contract.
    assert runtime["flags"]["device"] is None
    assert runtime["flags"]["flash_attn"] is True
    assert runtime["flags"]["jinja"] is False
    assert runtime["flags"]["spec"]["enabled"] is False
    # STACKCHG-DFLASH2-20261003: the stack manifest now declares worker_vision
    # tier WARM (cold CPU, not started by default), so its residency cost is the
    # warm cost (was 1.0 = hot while it was MI210-resident).
    assert role["priors"]["memory_cost"] == 2.0


def test_compile_refuses_missing_descriptor_without_allow_incomplete(tmp_path: Path) -> None:
    registry_path = _write_yaml(tmp_path / "registry.yaml", {"server_mode": {}, "roles": {}})
    descriptor_path = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})

    with pytest.raises(StackPriorsCompileError) as exc:
        compile_stack_priors(
            registry_path=registry_path,
            descriptor_path=descriptor_path,
            active_roles={_RETIRED_ARCHITECT_ROLE},
        )

    assert f"{_RETIRED_ARCHITECT_ROLE}: Missing model descriptor binding" in str(exc.value)


# --------------------------------------------------------------------------- #
# ESC-8 Fix 6: priors compile must not read the ambient default-full env.       #
# --------------------------------------------------------------------------- #

def _quarters_connect(_host: str, port: int) -> bool:
    """Quarters-only fleet: the full host ports are dead, everything else live.

    SSU-F5: the dead set is DERIVED from stack_topology's ``full_instance_idx``
    via the same helper the probe under test uses to decide what to probe. The
    literal ``{8070, 8072, 8085}`` that stood here was a third restatement of the
    lineup -- 8072 was worker_general's own full port and 8085 a retired
    ingest_long_context server, so this fixture was simulating a fleet that has
    not existed since 2026-09-22 while still passing.
    """
    from scripts.server.realized_fleet import full_instance_ports

    return port not in full_instance_ports()


def _fulls_connect(_host: str, _port: int) -> bool:
    return True


def _all_dead_connect(_host: str, _port: int) -> bool:
    return False


def test_realized_compile_numa_mode_env_unset_uses_realized() -> None:
    assert _realized_compile_numa_mode(environ={}, connect=_quarters_connect) == "quarter"
    assert _realized_compile_numa_mode(environ={}, connect=_fulls_connect) in {"full", "both"}


def test_realized_compile_numa_mode_env_contradiction_prefers_realized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        mode = _realized_compile_numa_mode(
            environ={"ORCHESTRATOR_STACK_NUMA_MODE": "full"},
            connect=_quarters_connect,
        )
    assert mode == "quarter"
    assert any("contradicts the realized fleet" in rec.message for rec in caplog.records)


def test_realized_compile_numa_mode_env_agreement_kept() -> None:
    assert (
        _realized_compile_numa_mode(
            environ={"ORCHESTRATOR_STACK_NUMA_MODE": "quarter"},
            connect=_quarters_connect,
        )
        == "quarter"
    )


def test_realized_compile_numa_mode_refuses_without_signal() -> None:
    with pytest.raises(StackPriorsModeError):
        _realized_compile_numa_mode(environ={}, connect=_all_dead_connect)
    # An explicit env does NOT rescue a no-signal probe (unverifiable fleet).
    with pytest.raises(StackPriorsModeError):
        _realized_compile_numa_mode(
            environ={"ORCHESTRATOR_STACK_NUMA_MODE": "full"},
            connect=_all_dead_connect,
        )


def _worker_math_conflict_paths(tmp_path: Path) -> tuple[Path, Path]:
    registry_path = _write_yaml(
        tmp_path / "registry.yaml",
        {"server_mode": {}, "roles": {"worker_math": {"memory": {"residency": "warm"}}}},
    )
    descriptor_path = _write_yaml(
        tmp_path / "descriptors.yaml",
        {
            "models": [
                {
                    "model_id": "qwen2.5-math-7b-q4",
                    "role_bindings": {"roles": ["worker_math"], "server_roles": []},
                    "quality": {"suite_vector": {}, "measured": []},
                    "speed": {"measured": []},
                    "acceleration": {},
                    "serving": {"ports": []},
                    "known_gaps": ["Role-server conflict: stale worker server binding"],
                }
            ]
        },
    )
    return registry_path, descriptor_path


def test_compile_require_realized_mode_derives_quarter_lineup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """V4: a clean-shell compile with a sub-full-live probe derives the sub-full
    lineup (alias inherits the host's sub-full fleet), never the dead full port.

    The sub-full shape is whatever the role declares in NUMA_CONFIG: quarters when
    this test was written, halves since the 2026-07-30 quarter retirement. The
    invariant under test — the dead full host port must never appear — is
    shape-independent."""
    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    registry_path, descriptor_path = _worker_math_conflict_paths(tmp_path)

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_math"},
        allow_incomplete=True,
        require_realized_mode=True,
        connect=_quarters_connect,
    )

    ports = priors["roles"]["worker_math"]["serving"]["ports"]
    # SSU-F5: the invariant is shape- AND lineup-independent, so both halves are
    # DERIVED. The dead FULL port of whichever host worker_math rides must not
    # appear, and the fleet must be exactly that host's sub-full siblings. The
    # literals were [8082, 8182] (2 halves on worker_general's own server) and,
    # before the 2026-07-30 quarter retirement, four quarters.
    for dead in _declared_fleet_ports("worker_math", "full"):
        assert dead not in ports, dead
    assert ports == _declared_fleet_ports("worker_math", "quarter")
    assert ports, "sub-full lineup resolved empty: the probe derived nothing"


def test_compile_require_realized_mode_refuses_without_signal(tmp_path: Path) -> None:
    """V5: a compile with no realized signal refuses rather than defaulting full."""
    registry_path, descriptor_path = _worker_math_conflict_paths(tmp_path)
    with pytest.raises(StackPriorsModeError):
        compile_stack_priors(
            registry_path=registry_path,
            descriptor_path=descriptor_path,
            active_roles={"worker_math"},
            allow_incomplete=True,
            require_realized_mode=True,
            connect=_all_dead_connect,
        )


def test_compile_default_does_not_probe_realized_fleet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The realized-fleet resolution is strictly opt-in: a plain compile never
    calls it (so it can neither refuse nor probe sockets by default)."""
    import src.registry.stack_priors as sp

    def _boom(**_kw):
        raise AssertionError("resolver must not run without require_realized_mode")

    monkeypatch.setattr(sp, "_realized_compile_numa_mode", _boom)
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
    registry_path, descriptor_path = _worker_math_conflict_paths(tmp_path)

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_math"},
        allow_incomplete=True,
    )
    # Legacy full-mode default is unchanged -- SSU-F5: DERIVED, so "full mode"
    # means the host's declared full instance rather than a port that happened to
    # be worker_general's own in 2026-08.
    assert priors["roles"]["worker_math"]["serving"]["ports"] == _declared_fleet_ports(
        "worker_math", "full"
    )


def test_policy_hints_returns_none_thresholds_when_model_memory_unknown() -> None:
    hints = _policy_hints({"launch": {"modes": ["default"], "entries": []}}, {})
    assert hints["lock_class"] == "exclusive"
    assert hints["contention_class"] == "heavy"
    assert hints["tap_safe_non_stream"] is None
    assert hints["high_cost"] is None
    assert hints["model_mem_gb"] is None
    assert hints["source"] == "stack_priors.compile"
    assert hints["thresholds"] == {
        "tap_safe_non_stream_min_mem_gb": 64.0,
        "high_cost_min_mem_gb": 60.0,
    }


def test_policy_hints_classify_shared_worker_as_light_and_low_cost() -> None:
    serving = {"launch": {"modes": ["worker_pool"], "entries": []}}
    hints = _policy_hints(serving, {"mem_gb": 37})
    assert hints["lock_class"] == "shared"
    assert hints["contention_class"] == "light"
    assert hints["tap_safe_non_stream"] is False
    assert hints["high_cost"] is False
    assert hints["model_mem_gb"] == 37.0


def test_policy_hints_flag_heavy_high_cost_role() -> None:
    serving = {"launch": {"modes": ["default"], "entries": []}}
    hints = _policy_hints(serving, {"mem_gb": 238})
    assert hints["lock_class"] == "exclusive"
    assert hints["contention_class"] == "heavy"
    assert hints["tap_safe_non_stream"] is True
    assert hints["high_cost"] is True


def test_compile_projects_ctx_model_max_and_policy_hints(tmp_path: Path) -> None:
    registry_path = _write_yaml(
        tmp_path / "registry.yaml",
        {
            "server_mode": {
                "worker": {
                    "url": "http://localhost:8072",
                    "port": 8072,
                    "tier": "hot",
                    "model_role": "worker_general",
                    "throughput": "60.7",
                }
            },
            "roles": {"worker_general": {"memory": {"residency": "warm"}}},
        },
    )
    descriptor_path = _write_yaml(
        tmp_path / "descriptors.yaml",
        {
            "models": [
                {
                    "model_id": "gemma4-26b-a4b-q4",
                    "mem_gb": 37,
                    "ctx_max": 16384,
                    "ctx_model_max": 131072,
                    "role_bindings": {
                        "roles": ["worker_general"],
                        "server_roles": ["worker"],
                    },
                    "quality": {"suite_vector": {"overall": 0.9}, "measured": []},
                    "speed": {"quarter_48t_tps": 60.7, "measured": []},
                    "acceleration": {"spec_type": "mtp"},
                    "serving": {"ports": [8072], "binary": "llama.cpp"},
                    "known_gaps": [],
                }
            ]
        },
    )

    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles={"worker_general"},
        allow_incomplete=True,
    )

    worker = priors["roles"]["worker_general"]
    # 620: model-native context is projected alongside the effective ctx_max.
    assert worker["model"]["ctx_model_max"] == 131072
    assert worker["model"]["ctx_max"] == 16384
    # Additive fields must not break the generated contract shape.
    assert validate_stack_priors_contract(priors) == []
    # 622: policy hints projected. lock/contention class is a function of the
    # LAUNCH MODE of the process the role rides -- a shared `worker_pool` launch
    # takes a shared lock and is light; every other launch is exclusive/heavy.
    # SSU-F5: that mode is DERIVED from role_launch_meta rather than pinned.
    # "shared"/"light" was correct while worker_general ran its own worker_pool
    # server; the 2026-09-22 lineup change moved it onto frontdoor's `default`
    # process, which flips both classes -- correctly, and without a test edit.
    shared_launch = _lineup_launch_mode("worker_general") == "worker_pool"
    policy = worker["policy"]
    assert policy["lock_class"] == ("shared" if shared_launch else "exclusive")
    assert policy["contention_class"] == ("light" if shared_launch else "heavy")
    assert policy["tap_safe_non_stream"] is False
    assert policy["high_cost"] is False
    assert policy["model_mem_gb"] == 37.0


# ── stack-change-kvu (2026-09-24): kv_unified is a DECLARED boolean ──────────
@pytest.mark.parametrize(
    ("server_cfg", "role_cfg", "expected"),
    [
        # serving_shape is the canonical home and outranks a flat copy.
        ({"serving_shape": {"kv_unified": True}, "kv_unified": False}, None, True),
        # False is a declaration, not an absence.
        ({"serving_shape": {"kv_unified": False}}, None, False),
        # Role-local serving block works too.
        (None, {"serving": {"kv_unified": True}}, True),
        # Undeclared -> None (launcher emits nothing).
        ({"serving_shape": {"n_ctx": 196608}}, {}, None),
        # A non-bool is not coerced (1 is not True here).
        ({"serving_shape": {"kv_unified": 1}}, None, None),
    ],
)
def test_runtime_flag_bool_prior_resolves_kv_unified(server_cfg, role_cfg, expected) -> None:
    from src.registry.stack_priors import _runtime_flag_bool_prior

    assert (
        _runtime_flag_bool_prior(server_cfg, role_cfg, key="kv_unified") is expected
    )


@pytest.mark.parametrize(
    ("server_cfg", "expected"),
    [
        ({"serving_shape": {"draft_kv_quant": {"k": "q8_0", "v": "q8_0"}}}, ("q8_0", "q8_0")),
        ({"serving_shape": {"draft_kv_quant": {"k": "q8_0"}}}, None),
        ({"serving_shape": {"kv_quant": {"k": "q8_0", "v": "q8_0"}}}, None),
        (None, None),
    ],
)
def test_draft_kv_types_prior_reads_only_serving_shape(server_cfg, expected) -> None:
    from src.registry.stack_priors import _draft_kv_types_prior

    assert _draft_kv_types_prior(server_cfg, None) == expected
