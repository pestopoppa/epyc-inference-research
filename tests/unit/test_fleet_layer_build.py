"""WP-12 fleet layer — fleet construction from the registry server_mode SoT.

Acceptance plan coverage (wp12-fleet-layer-design.md §6):
  * case 1 — fleet collapse / parity over the REAL registry + priors + NUMA
  * case 3 — no phantom-full: quarters-only realized set never yields a full
  * case 8 — remappability: re-pointing worker_math is a data-only change
  * case 9 — ESC-8 non-clobber: env NUMA mode cannot override fleet identity
plus the §3 parity invariant (fail closed) and the §8 degraded bootstrap
(one literal per FLEET, never per role).

All offline: synthetic dicts + tmp YAML files; no sockets, no inference.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import src.fleet as fleet_mod
from src.fleet import (
    FLEET_LAYER_ENV,
    FleetBuildError,
    FleetParityError,
    build_fleets_and_bindings,
    fleet_layer_enabled,
    resolve_binding,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_REGISTRY = REPO_ROOT / "orchestration" / "model_registry.yaml"
REAL_PRIORS = REPO_ROOT / "orchestration" / "derived" / "stack_priors.yaml"


# ── Expectations derived from the REAL artifacts, never restated ─────────────
#
# The "case 1" tests exist to check the builder against the production sources,
# so their expectations must come from those same sources. Restating them
# hardcoded the machine of the day: the port tuples pinned the 2026-07-23
# big+quarters lineup and went stale at the 2026-07-30 quarters retirement,
# and the frontdoor membership set pinned the pre-W1 alias roster and went
# stale when coder_escalation moved to architect_general's :8083 process.


def _real_server_mode() -> dict:
    return yaml.safe_load(REAL_REGISTRY.read_text(encoding="utf-8"))["server_mode"]


def _expected_bound_roles(server_mode: dict, fleet_id: str) -> set[str]:
    """Roles the REGISTRY declares on ``fleet_id``'s process.

    Both alias forms count: `shared_with` members of the fleet's own row, and
    any row that declares `alias_of` this fleet (the form the W1 cutover used).
    """
    from src.fleet import _canonical_role

    bound = {fleet_id}
    for name, row in server_mode.items():
        if not isinstance(row, dict):
            continue
        target = row.get("alias_of")
        if target:
            if _canonical_role(str(target)) == fleet_id:
                bound.add(_canonical_role(str(name)))
            continue
        if _canonical_role(str(name)) != fleet_id:
            continue
        for shared in row.get("shared_with") or []:
            bound.add(_canonical_role(str(shared)))
    return bound


def _config_alias_names() -> set[str]:
    """Config-layer role labels that have no registry row of their own."""
    from src.config import models as config_models

    return set(config_models._CANONICAL_SERVER_URL_ALIASES) | set(
        config_models._RUNTIME_SELECTED_ROLE_ALIASES
    )


def _real_host_fleet(server_mode: dict, role: str) -> str:
    """The server_mode row that physically HOSTS ``role``.

    2026-09-26: the operator-signed 2026-09-22 lineup cutover (orchestrator
    860b0b2d) made worker_general / worker_math / toolrunner / worker_explore
    aliases of frontdoor's :8070 CPU fleet, so there is no ``worker_general``
    fleet any more. The case-1 tests used to name that fleet literally; they now
    ask the registry which row hosts the worker roles (``shared_with`` on a
    non-alias row), so the next lineup change moves the expectation with it.
    """
    for host, row in server_mode.items():
        if not isinstance(row, dict) or row.get("alias_of"):
            continue
        if host == role or role in (row.get("shared_with") or []):
            return host
    raise AssertionError(f"registry declares no host fleet for {role!r}")


def _real_priors_ports(role: str) -> list[int]:
    data = yaml.safe_load(REAL_PRIORS.read_text(encoding="utf-8"))
    return sorted(data["roles"][role]["serving"]["ports"])


def _real_topology_ports(role: str) -> tuple[int, list[int]]:
    """(aligned-full port, all instance ports) for ``role`` from NUMA_CONFIG."""
    from scripts.server.stack_numa import NUMA_CONFIG

    cfg = NUMA_CONFIG[role]
    instances = cfg["instances"]
    return instances[cfg["full_instance_idx"]][1], [inst[1] for inst in instances]


# ── Synthetic topology (house pattern: worker_general-shaped fleet) ──────────

SYN_NUMA = {
    "worker_general": {
        "instances": [
            ("0-95", 9072, 96),
            ("0-23,96-119", 9082, 48),
            ("24-47,120-143", 9182, 48),
            ("48-71,144-167", 9282, 48),
            ("72-95,168-191", 9382, 48),
        ],
        "full_instance_idx": 0,
        "placement_policy": "full_disabled",
    },
    "frontdoor": {
        "instances": [
            ("0-47,96-143", 9070, 96),
            ("0-23,96-119", 9080, 48),
            ("24-47,120-143", 9180, 48),
            ("48-71,144-167", 9280, 48),
            ("72-95,168-191", 9380, 48),
        ],
        "full_instance_idx": 0,
        "placement_policy": "burst_prefer_split",
    },
    "architect_general": {
        "instances": [("0-95", 9083, 96)],
    },
}

SYN_SERVER_MODE = {
    "frontdoor": {
        "port": 9070,
        "model_role": "qwen_synth",
        "shared_with": ["coder_escalation", "worker_summarize"],
    },
    "worker": {
        "port": 9072,
        "model_role": "worker_general",
        "shared_with": ["worker_math", "toolrunner"],
    },
    "architect_general": {
        "port": 9083,
        "model_role": "arch_synth",
    },
}


def _write_priors(tmp_path: Path, ports_by_role: dict[str, list[int]]) -> Path:
    payload = {
        "roles": {
            role: {
                "deployment_status": "live_stack",
                "serving": {"ports": list(ports)},
            }
            for role, ports in ports_by_role.items()
        }
    }
    path = tmp_path / "stack_priors.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


QUARTER_PORTS = [9082, 9182, 9282, 9382]


def _build_synthetic(tmp_path, *, ports=None, server_mode=None, numa=None):
    priors = _write_priors(
        tmp_path,
        ports if ports is not None else {
            "worker_general": QUARTER_PORTS,
            "worker_math": QUARTER_PORTS,
            "toolrunner": QUARTER_PORTS,
            "frontdoor": [9080, 9180, 9280, 9380],
            "architect_general": [9083],
        },
    )
    return build_fleets_and_bindings(
        registry_server_mode=server_mode if server_mode is not None else SYN_SERVER_MODE,
        numa_config=numa if numa is not None else SYN_NUMA,
        priors_path=priors,
    )


# ── Flag default ─────────────────────────────────────────────────────────────


def test_fleet_layer_flag_default_off(monkeypatch):
    monkeypatch.delenv(FLEET_LAYER_ENV, raising=False)
    assert fleet_layer_enabled() is False
    monkeypatch.setenv(FLEET_LAYER_ENV, "1")
    assert fleet_layer_enabled() is True
    monkeypatch.setenv(FLEET_LAYER_ENV, "0")
    assert fleet_layer_enabled() is False


# ── Case 1 — fleet collapse / parity over the REAL artifacts ─────────────────


def test_case1_real_registry_collapses_shared_roles_to_one_fleet():
    from src.fleet import _canonical_role

    server_mode = _real_server_mode()
    fleets, bindings = build_fleets_and_bindings(
        registry_server_mode=server_mode,
        priors_path=REAL_PRIORS,
    )

    worker_host = _real_host_fleet(server_mode, "worker_general")
    assert worker_host in fleets
    assert "frontdoor" in fleets

    # Membership is exactly what the registry declares, for EVERY fleet.
    for fleet_id, fleet in fleets.items():
        assert set(fleet.bound_roles) == _expected_bound_roles(server_mode, fleet_id), (
            fleet_id
        )

    # Every declared member — plus the config-layer aliases that have no
    # registry row of their own (worker, coder, worker_explore, ...) — resolves
    # to the IDENTICAL endpoint tuple + topology_role (the §3 invariant).
    alias_names = _config_alias_names()
    for fleet_id, fleet in fleets.items():
        members = set(fleet.bound_roles)
        members |= {a for a in alias_names if _canonical_role(a) in members}
        for role in sorted(members):
            binding = resolve_binding(role, bindings)
            assert binding is not None, role
            assert binding.fleet_id == fleet_id, role
            assert fleets[binding.fleet_id].endpoints == fleet.endpoints
            assert fleets[binding.fleet_id].topology_role == fleet_id

    # Non-vacuity: the collapse must actually collapse several roles onto one
    # fleet, or the loop above would pass over singletons.
    # The worker roles collapse onto their registry-declared host (frontdoor's
    # :8070 fleet since the 2026-09-22 cutover), whatever that host is.
    assert {"worker_general", "worker_math", "toolrunner"} <= set(
        fleets[worker_host].bound_roles
    )
    assert "worker_summarize" in fleets["frontdoor"].bound_roles
    assert resolve_binding("worker_explore", bindings).fleet_id == worker_host

    # W1 cutover: coder_escalation is declared `alias_of: architect_general`, so
    # it rides THAT process — it is neither a fleet of its own (the phantom the
    # builder used to create, which then tripped the double-binding guard) nor a
    # frontdoor member any more.
    assert "coder_escalation" not in fleets
    assert "coder_escalation" not in fleets["frontdoor"].bound_roles
    ce = resolve_binding("coder_escalation", bindings)
    # 2026-09-27 ARCHITECT SWAP: the :8083 27B host row is architect_critic.
    assert ce is not None and ce.fleet_id == "architect_critic"
    assert resolve_binding("coder", bindings).fleet_id == "architect_critic"

    # worker_fast is a DISTINCT physical server, never a worker alias.
    wf = resolve_binding("worker_fast", bindings)
    assert wf is None or wf.fleet_id != worker_host


def test_case1_real_worker_fleet_realizes_full_plus_quarters():
    """The worker fleet realizes exactly what the CHECKED-IN priors declare:
    the aligned idx-0 full plus every sibling instance, in mixed mode. Ports are
    read from the priors artifact under test (cross-checked against NUMA_CONFIG)
    rather than restated — the literal that stood here pinned the 2026-07-23
    big+quarters lineup and survived the 2026-07-30 retirement of two of those
    ports. (The quarters-only shape stays covered by the synthetic case-3
    fixtures.)

    Since the 2026-09-22 cutover the worker roles have no fleet of their own:
    the fleet under test is the registry-declared HOST of worker_general, and
    its NUMA_CONFIG entry is keyed by that host (the worker_general entry was
    deleted)."""
    server_mode = _real_server_mode()
    fleets, _ = build_fleets_and_bindings(
        registry_server_mode=server_mode,
        priors_path=REAL_PRIORS,
    )
    worker_host = _real_host_fleet(server_mode, "worker_general")
    worker_fleet = fleets[worker_host]
    # The priors carry a row for the ALIAS too; it must agree with its host's.
    expected_ports = _real_priors_ports("worker_general")
    assert expected_ports == _real_priors_ports(worker_host)
    full_port, topology_ports = _real_topology_ports(worker_host)

    assert sorted(worker_fleet.ports) == expected_ports
    # The priors and the topology must describe the same machine.
    assert expected_ports == sorted(topology_ports)
    assert worker_fleet.full_endpoint is not None
    assert worker_fleet.full_endpoint.port == full_port
    assert worker_fleet.mode == "mixed"
    assert not worker_fleet.degraded
    # Mixed mode means full + at least one sibling; a 1-endpoint fleet would
    # make the mode assertion meaningless.
    assert len(expected_ports) >= 2


# ── Case 3 — no phantom-full ─────────────────────────────────────────────────


def test_case3_quarters_only_fleet_yields_no_full_endpoint(tmp_path):
    """A quarters-only realized set (whose first port the priors URL serializer
    would have mislabeled ``full:``) produces quarter endpoints at their TRUE
    port-resolved topology idxs and no full — the DISPATCH-A2 demotion done
    once, structurally, at fleet build."""
    fleets, _ = _build_synthetic(tmp_path)
    wf = fleets["worker_general"]

    assert wf.full_endpoint is None
    assert wf.mode == "quarter"
    assert [ep.port for ep in wf.quarter_endpoints] == QUARTER_PORTS
    assert [ep.topology_idx for ep in wf.quarter_endpoints] == [1, 2, 3, 4]
    # Region locks == physical cores: each endpoint's region set matches its
    # NUMA_CONFIG cpuset, and no endpoint holds the all-region (idx-0) shape.
    from src.runtime.instance_topology import cpu_list_to_regions

    for ep in wf.quarter_endpoints:
        expected = cpu_list_to_regions(
            SYN_NUMA["worker_general"]["instances"][ep.topology_idx][0]
        )
        assert ep.regions == expected
        assert len(ep.regions) == 1

    # And the config-compatible URL value never advertises a phantom full.
    assert "full:" not in wf.url_value


def test_aligned_full_is_recognized(tmp_path):
    fleets, _ = _build_synthetic(
        tmp_path,
        ports={"worker_general": [9072] + QUARTER_PORTS, "frontdoor": [9080]},
    )
    wf = fleets["worker_general"]
    assert wf.full_endpoint is not None
    assert wf.full_endpoint.port == 9072
    assert wf.full_endpoint.topology_idx == 0
    assert wf.mode == "mixed"
    assert wf.url_value.startswith("full:http://localhost:9072,")


# ── Case 8 — remappability ───────────────────────────────────────────────────


def test_case8_repointing_worker_math_is_a_data_only_change(tmp_path):
    """Re-point worker_math at its own (synthetic) fleet purely by editing the
    server_mode + priors DATA. Only worker_math moves; worker_general and
    toolrunner stay; no code/URL-literal edit involved."""
    remapped_server_mode = {
        "frontdoor": SYN_SERVER_MODE["frontdoor"],
        "worker": {
            "port": 9072,
            "model_role": "worker_general",
            "shared_with": ["toolrunner"],  # worker_math removed
        },
        "worker_math": {
            "port": 9099,
            "model_role": "qwen25_math_ghost",
        },
        "architect_general": SYN_SERVER_MODE["architect_general"],
    }
    ports = {
        "worker_general": QUARTER_PORTS,
        "toolrunner": QUARTER_PORTS,
        "worker_math": [9099],
        "frontdoor": [9080, 9180, 9280, 9380],
        "architect_general": [9083],
    }
    fleets, bindings = _build_synthetic(
        tmp_path, ports=ports, server_mode=remapped_server_mode
    )

    assert bindings["worker_math"].fleet_id == "worker_math"
    assert bindings["worker_math"].model_binding == "qwen25_math_ghost"
    assert fleets["worker_math"].ports == (9099,)

    assert bindings["worker_general"].fleet_id == "worker_general"
    assert bindings["toolrunner"].fleet_id == "worker_general"
    assert sorted(fleets["worker_general"].ports) == QUARTER_PORTS


# ── Case 9 — ESC-8 non-clobber ───────────────────────────────────────────────


def test_case9_env_numa_mode_cannot_override_fleet_identity(tmp_path, monkeypatch):
    """With ORCHESTRATOR_STACK_NUMA_MODE=full set but a quarters-only priors
    artifact, the fleet realizes the quarter ports — the env producer is
    structurally not consulted for fleet identity (design §2.1)."""
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
    fleets, _ = _build_synthetic(tmp_path)
    wf = fleets["worker_general"]
    assert sorted(wf.ports) == QUARTER_PORTS
    assert wf.full_endpoint is None
    assert wf.mode == "quarter"


# ── §3 parity invariant — fail closed ────────────────────────────────────────


def test_parity_violation_fails_closed(tmp_path):
    """A bound role whose priors record names a DIFFERENT endpoint set than its
    fleet (the worker_math stale-copy incident class) refuses the build."""
    ports = {
        "worker_general": QUARTER_PORTS,
        "worker_math": [9082, 9182],  # stale 2-endpoint copy
        "frontdoor": [9080, 9180, 9280, 9380],
        "architect_general": [9083],
    }
    with pytest.raises(FleetParityError, match="worker_math"):
        _build_synthetic(tmp_path, ports=ports)


def test_role_bound_to_two_fleets_fails_closed(tmp_path):
    server_mode = {
        "frontdoor": {
            "port": 9070,
            "shared_with": ["worker_math"],
        },
        "worker": {
            "port": 9072,
            "model_role": "worker_general",
            "shared_with": ["worker_math"],
        },
    }
    ports = {
        "worker_general": QUARTER_PORTS,
        "frontdoor": [9080, 9180, 9280, 9380],
    }
    with pytest.raises(FleetBuildError, match="worker_math"):
        _build_synthetic(tmp_path, ports=ports, server_mode=server_mode)


# ── §8 degraded bootstrap — one literal per FLEET ────────────────────────────


def test_degraded_bootstrap_uses_per_fleet_literal(tmp_path):
    """Priors absent (fresh clone / pre-launch API): each fleet resolves its
    single per-fleet literal; roles still carry NO private copies."""
    server_mode = yaml.safe_load(REAL_REGISTRY.read_text(encoding="utf-8"))["server_mode"]
    fleets, bindings = build_fleets_and_bindings(
        registry_server_mode=server_mode,
        priors_path=tmp_path / "missing.yaml",
    )
    # The worker roles' registry-declared host fleet (frontdoor since the
    # 2026-09-22 cutover), not a literal fleet name.
    worker_host = _real_host_fleet(server_mode, "worker_general")
    wf = fleets[worker_host]
    full_port, topology_ports = _real_topology_ports(worker_host)
    assert wf.degraded
    # `_derive_degraded_fallback_ports` reads NUMA_CONFIG (2026-07-30: "fleet.py
    # derives degraded-fallback ports instead of hardcoding retired ones"), so
    # the expectation is derived from that upstream topology artifact — not from
    # the derivation function itself, which would be tautological. A regression
    # that decoupled the fallback from the topology still fails here.
    assert sorted(wf.ports) == sorted(topology_ports)
    # The literal resolves through the same port→topology alignment: the idx-0
    # port IS the true full for the worker host in the real NUMA_CONFIG.
    assert wf.full_endpoint is not None and wf.full_endpoint.port == full_port
    # Non-vacuity: the degraded literal must be a real multi-instance fleet.
    assert len(topology_ports) >= 2
    # Shared roles reference the fleet — no per-role literals resurface.
    assert resolve_binding("worker_general", bindings).fleet_id == worker_host
    assert resolve_binding("worker_math", bindings).fleet_id == worker_host
    assert resolve_binding("toolrunner", bindings).fleet_id == worker_host


# ── Cached accessor fail-safe ────────────────────────────────────────────────


def test_get_fleets_and_bindings_latches_failure_and_resets(monkeypatch):
    calls = {"n": 0}

    def _boom(**_kw):
        calls["n"] += 1
        raise FleetBuildError("synthetic failure")

    monkeypatch.setattr(fleet_mod, "build_fleets_and_bindings", _boom)
    fleet_mod.reset_fleet_cache()
    try:
        assert fleet_mod.get_fleets_and_bindings() is None
        assert fleet_mod.get_fleets_and_bindings() is None
        # Failure is latched — the broken build is not retried per call.
        assert calls["n"] == 1
        fleet_mod.reset_fleet_cache()
        assert fleet_mod.get_fleets_and_bindings() is None
        assert calls["n"] == 2
    finally:
        fleet_mod.reset_fleet_cache()
