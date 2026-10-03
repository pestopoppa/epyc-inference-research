"""Additional unit tests for seeding_types state/fallback branches."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark"
sys.path.insert(0, str(_ROOT))
_SPEC = importlib.util.spec_from_file_location("seeding_types_state_test", _ROOT / "seeding_types.py")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["seeding_types_state_test"] = _MOD
_SPEC.loader.exec_module(_MOD)

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"


def test_read_registry_timeout_returns_fallback_when_registry_unreadable():
    with patch("pathlib.Path.open", side_effect=OSError("boom")):
        assert _MOD._read_registry_timeout("benchmark", "seeding_default", 600) == 600


def test_read_stack_prior_default_roles_filters_live_seedable_roles(tmp_path: Path):
    stack_priors = tmp_path / "stack_priors.yaml"
    stack_priors.write_text(
        """
roles:
  architect_general:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8083
  reap_25b_frontdoor:
    deployment_status: benchmark_or_candidate
    serving:
      endpoint: http://localhost:8090
  toolrunner:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8072
  worker_general:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8072
""",
        encoding="utf-8",
    )

    roles = _MOD._read_stack_prior_default_roles(stack_priors)

    assert roles == ["architect_general", "worker_general"]


def test_read_stack_prior_active_roles_includes_live_vision_roles(
    tmp_path: Path,
) -> None:
    stack_priors = tmp_path / "stack_priors.yaml"
    stack_priors.write_text(
        """
roles:
  frontdoor:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8070
      ports: [8070]
      server_role: frontdoor
    priors:
      memory_cost: 1.0
  toolrunner:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8072
      ports: [8072]
      server_role: worker
    priors:
      memory_cost: 1.0
  worker_vision:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8086
      ports: [8086]
      server_role: worker_vision
    priors:
      memory_cost: 1.0
  vision_escalation:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8087
      ports: [8087]
      server_role: vision_escalation
    priors:
      memory_cost: 1.0
  reap_25b_frontdoor:
    deployment_status: benchmark_or_candidate
    serving:
      endpoint: http://localhost:8090
      ports: [8090]
""",
        encoding="utf-8",
    )

    # The fixture declares no model.mem_gb, so heaviness resolves through the
    # port fallback. Declare the heavy topology this assertion is about instead
    # of inheriting whatever the live machine happens to be running.
    # Likewise the cost tiers: with no mem_gb and memory_cost 1.0 they resolve
    # through ROLE_COST_TIER, which seeding_types overlays at import time with
    # tiers derived from the repo's lean registry. Pin the tiers this ordering
    # assertion is about instead of inheriting the current lineup's (the lean's
    # worker_vision memory_gb went 0 -> 18.3 when STACKCHG-DFLASH2-20261003 moved
    # the VL-30B to CPU, which re-tiers it 1 -> 2 and silently reorders the result).
    fixture_tiers = {"worker_vision": 1, "frontdoor": 2, "vision_escalation": 3}
    with patch.object(_MOD, "HEAVY_PORTS", {8087}), \
         patch.dict(_MOD.ROLE_COST_TIER, fixture_tiers, clear=True):
        roles = _MOD._read_stack_prior_active_roles(stack_priors)

    by_name = {role["name"]: role for role in roles}
    assert list(by_name) == ["worker_vision", "frontdoor", "vision_escalation"]
    assert {name: role["cost_tier"] for name, role in by_name.items()} == fixture_tiers
    assert by_name["worker_vision"]["port"] == 8086
    assert by_name["vision_escalation"]["is_heavy"] is True
    assert by_name["vision_escalation"]["cost_tier"] == 3
    assert "toolrunner" not in by_name
    assert "reap_25b_frontdoor" not in by_name


def test_stack_prior_cost_tier_prefers_model_memory_over_stale_static_tiers(
    tmp_path: Path,
) -> None:
    stack_priors = tmp_path / "stack_priors.yaml"
    stack_priors.write_text(
        """
roles:
  coder_escalation:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8084
      ports: [8084]
      server_role: coder_escalation
    priors:
      memory_cost: 1.0
    model:
      mem_gb: 37.0
  vision_escalation:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8087
      ports: [8087]
      server_role: vision_escalation
    priors:
      memory_cost: 1.0
    model:
      mem_gb: 18.0
  architect_general:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8083
      ports: [8083]
      server_role: architect_general
    priors:
      memory_cost: 1.0
    model:
      mem_gb: 69.0
""",
        encoding="utf-8",
    )

    roles = _MOD._read_stack_prior_active_roles(stack_priors)

    by_name = {role["name"]: role for role in roles}
    assert by_name["vision_escalation"]["cost_tier"] == 1
    assert by_name["coder_escalation"]["cost_tier"] == 2
    assert by_name["architect_general"]["cost_tier"] == 4


def test_stack_prior_cost_tier_keeps_memory_cost_fallback_without_model_mem() -> None:
    assert _MOD._cost_tier_from_stack_priors(
        "vision_escalation",
        {"priors": {"memory_cost": 1.0}},
    ) == 3
    assert _MOD._cost_tier_from_stack_priors(
        "worker_general",
        {"priors": {"memory_cost": 2.0}},
    ) == 3
    assert _MOD._cost_tier_from_stack_priors(
        "unknown_role",
        {"priors": {"memory_cost": "not-a-number"}},
    ) == 3


def test_read_stack_prior_active_roles_canonicalizes_worker_explore(
    tmp_path: Path,
) -> None:
    stack_priors = tmp_path / "stack_priors.yaml"
    stack_priors.write_text(
        """
roles:
  worker_explore:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8072
      ports: [8072]
      server_role: worker_explore
    priors:
      memory_cost: 1.0
""",
        encoding="utf-8",
    )

    with patch.object(_MOD, "_read_registry_timeout", return_value=240):
        roles = _MOD._read_stack_prior_active_roles(stack_priors)

    assert roles == [
        {
            "name": "worker_general",
            "registry_key": "worker_explore",
            "model_role": "worker_general",
            "port": 8072,
            "is_heavy": False,
            "cost_tier": 1,
            "timeout_s": 240,
        }
    ]


def test_read_stack_prior_topology_derives_primary_model_ports(tmp_path: Path) -> None:
    stack_priors = tmp_path / "stack_priors.yaml"
    stack_priors.write_text(
        """
roles:
  frontdoor:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8070
      ports: [8070, 8080, 8180]
    model:
      mem_gb: 37.0
  worker_general:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8072
      ports: [8072, 8082]
    model:
      mem_gb: 16.0
  vision_escalation:
    deployment_status: live_stack
    serving:
      ports: [8087, 8187]
      launch:
        entries:
          - port: 8187
            alias: true
          - port: 8087
            alias: false
    model:
      mem_gb: 18.0
  candidate:
    deployment_status: benchmark_or_candidate
    serving:
      endpoint: http://localhost:8099
    model:
      mem_gb: 999.0
""",
        encoding="utf-8",
    )

    topology = _MOD._read_stack_prior_topology(stack_priors)

    assert topology["role_port"] == {
        "frontdoor": 8070,
        "vision_escalation": 8087,
        "worker_general": 8072,
    }
    assert topology["model_ports"] == [8070, 8072, 8087]
    assert topology["heavy_ports"] == {8070, 8087}


def test_discover_active_roles_prefers_stack_priors_over_registry(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "model_registry.yaml"
    registry_path.write_text("server_mode: {}\n", encoding="utf-8")
    stack_prior_roles = [
        {
            "name": "worker_vision",
            "registry_key": "worker_vision",
            "model_role": "worker_vision",
            "port": 8086,
            "is_heavy": False,
            "cost_tier": 1,
            "timeout_s": 60,
        }
    ]

    with patch.object(_MOD, "_read_stack_prior_active_roles", return_value=stack_prior_roles):
        assert _MOD.discover_active_roles() == stack_prior_roles

    assert _MOD.discover_active_roles(registry_path=registry_path) == []


def test_default_roles_and_architect_roles_exclude_retired_architect_role():
    assert _RETIRED_ARCHITECT_ROLE not in _MOD.DEFAULT_ROLES
    assert _RETIRED_ARCHITECT_ROLE not in _MOD.ARCHITECT_ROLES
    assert _RETIRED_ARCHITECT_ROLE not in _MOD.ROLE_COST_TIER

    # Derive the expected roster from the same artifact DEFAULT_ROLES is
    # compiled from (live, seedable stack-prior roles) rather than restating a
    # lineup snapshot: the architect count moves whenever the fleet changes,
    # but the derivation must not.
    live_architect_roles = {
        role for role in _MOD._read_stack_prior_default_roles() if role.startswith("architect_")
    }
    assert live_architect_roles, "live stack priors declare no architect_* role"
    assert _MOD.ARCHITECT_ROLES == live_architect_roles


def test_discover_default_roles_fallback_prefers_active_discovery(monkeypatch):
    monkeypatch.setattr(
        _MOD,
        "discover_active_roles",
        lambda registry_path=None: [
            {"name": "worker_general"},
            {"name": "worker_math"},
            {"name": "toolrunner"},
        ],
    )

    assert _MOD._discover_default_roles_fallback() == ["worker_general"]


def test_discover_default_roles_fallback_uses_emergency_role_when_empty(monkeypatch):
    monkeypatch.setattr(_MOD, "discover_active_roles", lambda registry_path=None: [])

    assert _MOD._discover_default_roles_fallback() == ["frontdoor"]


def test_read_registry_topology_derives_degraded_ports_and_heavy_set(tmp_path: Path):
    registry_path = tmp_path / "model_registry.yaml"
    registry_path.write_text(
        """
server_mode:
  frontdoor:
    model: frontdoor.gguf
    port: 8070
    memory_gb: 37
  worker:
    model: worker.gguf
    port: 8072
    memory_gb: 16
  architect_general:
    model: architect.gguf
    port: 8083
    memory_gb: 69
  voice_server:
    model_type: whisper
    port: 8099
""",
        encoding="utf-8",
    )

    topology = _MOD._read_registry_topology(registry_path)

    assert topology["role_port"] == {
        "architect_general": 8083,
        "frontdoor": 8070,
        "worker_general": 8072,
    }
    assert topology["model_ports"] == [8070, 8072, 8083]
    assert topology["heavy_ports"] == {8070, 8083}
    assert topology["role_cost_tier"] == {
        "architect_general": 4,
        "frontdoor": 2,
        "worker_general": 1,
    }


def test_discover_active_roles_includes_registry_role_timeouts(tmp_path: Path):
    registry_path = tmp_path / "model_registry.yaml"
    registry_path.write_text(
        """
runtime_defaults:
  timeouts:
    default: 600
    roles:
      frontdoor: 180
      worker: 240
server_mode:
  frontdoor:
    model: frontdoor.gguf
    port: 8070
  worker:
    model: worker.gguf
    port: 8072
  voice_server:
    model_type: whisper
    port: 8099
""",
        encoding="utf-8",
    )

    roles = _MOD.discover_active_roles(registry_path=registry_path)

    by_name = {role["name"]: role for role in roles}
    assert by_name["frontdoor"]["timeout_s"] == 180
    assert by_name["worker_general"]["timeout_s"] == 240
    assert by_name["worker_general"]["cost_tier"] == 1
    assert "voice_server" not in by_name


def test_state_get_poll_client_lazily_creates_and_reuses_httpx_client():
    created = []
    fake_client = object()

    def _client_ctor(timeout):  # noqa: ANN001
        created.append(timeout)
        return fake_client

    fake_httpx = ModuleType("httpx")
    fake_httpx.Client = _client_ctor

    prev_httpx = sys.modules.get("httpx")
    _MOD.state._poll_client = None
    sys.modules["httpx"] = fake_httpx
    try:
        c1 = _MOD.state.get_poll_client()
        c2 = _MOD.state.get_poll_client()
    finally:
        if prev_httpx is None:
            sys.modules.pop("httpx", None)
        else:
            sys.modules["httpx"] = prev_httpx
        _MOD.state._poll_client = None

    assert c1 is fake_client
    assert c2 is fake_client
    assert created == [10]


def test_state_close_poll_client_swallows_close_exception_and_clears_client():
    bad_client = SimpleNamespace(close=lambda: (_ for _ in ()).throw(RuntimeError("close failed")))
    _MOD.state._poll_client = bad_client
    _MOD.state.close_poll_client()
    assert _MOD.state._poll_client is None
