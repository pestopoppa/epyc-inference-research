"""Tests for `src/registry/registry_validator.py`.

The module had NO test file before 2026-08-12. These tests were added with the
`numa_ports` x `NUMA_CONFIG` cross-check (handoff P1-5), and they also pin the
three pre-existing checks so the new one cannot be traded against them.

Two things these tests deliberately do NOT do:

* They do not assert against a hand-written registry that only contains the
  roles under test. Every cross-check test starts from the REAL compiled lean
  registry (`orchestration/model_registry.yaml`) — the exact file
  `stack_commands.cmd_start` hands to `validate_or_raise` — and mutates one
  field. A synthetic 2-role fixture would let a check pass because its input
  was empty of everything it should have objected to.
* They do not exercise a private helper only. `test_live_registry_*` and the
  mutation tests all go through `validate_all` / `validate_or_raise`, the two
  entry points the launcher actually calls.
"""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest
import yaml

from src.registry import registry_validator as rv
from src.registry.registry_validator import (
    RegistryValidationError,
    validate_all,
    validate_or_raise,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The path `scripts/server/stack_commands.py:cmd_start` validates. Not a
# re-derivation: `test_consumer_validates_this_exact_registry` asserts the
# consumer's own source computes the same file.
LIVE_REGISTRY = _REPO_ROOT / "orchestration" / "model_registry.yaml"


# ---------------------------------------------------------------------------
# Fixtures — with explicit non-emptiness assertions
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def numa_config() -> dict:
    """The live declared NUMA topology, as the launcher imports it."""
    cfg = rv._load_numa_config()
    assert cfg, "topology fixture is EMPTY — every cross-check would pass vacuously"
    # Non-emptiness of the specific shape the tests rely on: at least one role
    # with a real multi-instance fleet, and the full-instance convention in use.
    multi = {role for role, c in cfg.items() if len(c.get("instances") or []) > 1}
    assert multi, f"no multi-instance role in topology; roles={sorted(cfg)}"
    assert "frontdoor" in multi, f"expected frontdoor fleet; multi-instance roles={sorted(multi)}"
    assert cfg["frontdoor"].get("full_instance_idx") == 0
    assert [inst[1] for inst in cfg["frontdoor"]["instances"]] == [8070, 8080, 8180]
    return cfg


@pytest.fixture(scope="module")
def live_registry() -> dict:
    """The real compiled lean registry, parsed."""
    doc = yaml.safe_load(LIVE_REGISTRY.read_text())
    assert isinstance(doc, dict) and doc, f"{LIVE_REGISTRY} did not parse to a non-empty mapping"
    server_mode = doc.get("server_mode") or {}
    rows = {k: v for k, v in server_mode.items() if isinstance(v, dict)}
    # INPUT-EMPTINESS GUARD: the cross-check iterates server_mode rows. If this
    # fixture had no rows — or no rows declaring numa_ports — every mutation
    # test below would "pass" while checking nothing.
    assert len(rows) >= 5, f"expected the real role set, got {sorted(rows)}"
    declaring = {k: v["numa_ports"] for k, v in rows.items() if "numa_ports" in v}
    assert declaring, "no server_mode row declares numa_ports — fixture is not the real registry"
    assert declaring["frontdoor"] == [8080, 8180], declaring
    assert rows["frontdoor"]["port"] == 8070
    # 2026-09-26: this fixture used to pin `worker.model_role == worker_general`
    # and `worker.numa_ports == [8082, 8182]` — a separate worker fleet. The
    # operator-signed 2026-09-22 cutover (orchestrator 860b0b2d) made `worker`
    # an `alias_of: frontdoor` row with no fleet of its own and deleted the
    # worker_general NUMA_CONFIG entry, so no live row binds by model_role any
    # more. The model_role-binding test now builds its row from the live
    # topology instead (see `_model_role_bound_row`). What the mutation tests
    # DO need from the live file is a multi-instance fleet declaration:
    assert any(len(ports) >= 2 for ports in declaring.values()), declaring
    return doc


def _multi_instance_topology_role(numa_config: dict) -> str:
    """A live topology role with a real multi-instance fleet (derived)."""
    multi = sorted(r for r, c in numa_config.items() if len(c.get("instances") or []) > 1)
    assert multi, f"no multi-instance role in topology; roles={sorted(numa_config)}"
    return multi[0]


def _write(tmp_path: Path, doc: dict, name: str = "model_registry.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _numa_errors(errors: list[str]) -> list[str]:
    return [e for e in errors if "numa" in e.lower() or "topology" in e.lower()]


# ---------------------------------------------------------------------------
# The live registry must pass — through the real entry points
# ---------------------------------------------------------------------------


def test_live_registry_passes_validate_all() -> None:
    errors = validate_all(LIVE_REGISTRY)
    assert errors == [], errors


def test_live_registry_passes_validate_or_raise() -> None:
    """The exact call `stack_commands.cmd_start` makes."""
    validate_or_raise(LIVE_REGISTRY)


def test_round_tripped_live_registry_still_passes(tmp_path: Path, live_registry: dict) -> None:
    """The un-mutated round trip is the CONTROL for every mutation below.

    If yaml round-tripping alone produced errors, a mutation test could pass
    for a reason that has nothing to do with its mutation.
    """
    path = _write(tmp_path, copy.deepcopy(live_registry))
    assert validate_all(path) == []


# ---------------------------------------------------------------------------
# Mutations — each must FAIL, and fail for its own reason
# ---------------------------------------------------------------------------


def test_numa_ports_disagreeing_with_topology_is_rejected(
    tmp_path: Path, live_registry: dict, numa_config: dict
) -> None:
    doc = copy.deepcopy(live_registry)
    assert doc["server_mode"]["frontdoor"]["numa_ports"] == [8080, 8180]
    doc["server_mode"]["frontdoor"]["numa_ports"] = [8081, 8181]  # off-by-one drift
    errors = validate_all(_write(tmp_path, doc))
    assert any(
        "frontdoor" in e and "8081" in e and "disagrees with the NUMA topology" in e
        for e in errors
    ), errors


def test_numa_ports_drift_on_a_model_role_bound_row_is_rejected(
    tmp_path: Path, live_registry: dict, numa_config: dict
) -> None:
    """A row whose NAME is not a topology role binds via `model_role`.

    A name-equality-only check would silently skip such a row. Until the
    2026-09-22 cutover the live `worker` row was that row (model_role
    worker_general); since then no live row binds by model_role, so the row is
    built here from the live topology: a registry-only name whose model_role is
    a real multi-instance topology role. The correctly-projected declaration is
    the CONTROL (passes); one drifted port must then be rejected.
    """
    topo_role = _multi_instance_topology_role(numa_config)
    expected = rv._topology_fleet_ports(numa_config[topo_role])
    assert len(expected) >= 1, (topo_role, expected)
    row_name = "model_role_bound_row"
    assert row_name not in numa_config and row_name not in live_registry["server_mode"]

    doc = copy.deepcopy(live_registry)
    doc["server_mode"][row_name] = {"model_role": topo_role, "numa_ports": list(expected)}
    assert rv._bind_role_to_topology(row_name, doc["server_mode"][row_name], numa_config) == topo_role
    assert validate_all(_write(tmp_path, doc)) == [], "control: exact projection must pass"

    drifted = list(expected)
    drifted[-1] += 1
    doc["server_mode"][row_name]["numa_ports"] = drifted
    errors = validate_all(_write(tmp_path, doc))
    assert any(
        f"'{row_name}'" in e and topo_role in e and "disagrees" in e for e in errors
    ), errors


def test_phantom_fleet_for_an_unknown_role_is_rejected(
    tmp_path: Path, live_registry: dict, numa_config: dict
) -> None:
    """The blind spot in `stack_manifest.validate_declaration_parity()`.

    That guard iterates NUMA_CONFIG, so a registry role the topology has never
    heard of is skipped entirely. This is the real `vision_escalation`
    5-port phantom fleet (P1-7).
    """
    assert "vision_escalation" not in numa_config
    doc = copy.deepcopy(live_registry)
    assert "vision_escalation" not in doc["server_mode"]
    doc["server_mode"]["vision_escalation"] = {
        "url": "http://localhost:8087",
        "port": 8087,
        "numa_instances": 4,
        "numa_ports": [8187, 8287, 8387, 8487],
    }
    errors = validate_all(_write(tmp_path, doc))
    assert any(
        "vision_escalation" in e and "phantom" in e for e in errors
    ), errors


def test_numa_instances_disagreeing_with_numa_ports_length_is_rejected(
    tmp_path: Path, live_registry: dict
) -> None:
    # The row under test is derived: the first live row that declares a
    # consistent multi-port fleet (it was ingest_long_context until the
    # 2026-09-22 cutover made that an alias of architect_general's GPU process).
    doc = copy.deepcopy(live_registry)
    rows = doc["server_mode"]
    candidates = sorted(
        r for r, c in rows.items()
        if isinstance(c, dict)
        and isinstance(c.get("numa_ports"), list)
        and len(c["numa_ports"]) >= 2
        and c.get("numa_instances") == len(c["numa_ports"])
    )
    assert candidates, "no live row declares a consistent multi-port fleet"
    role = candidates[0]
    wrong = len(rows[role]["numa_ports"]) + 2
    rows[role]["numa_instances"] = wrong
    errors = validate_all(_write(tmp_path, doc))
    assert any(
        role in e and f"numa_instances={wrong}" in e for e in errors
    ), errors


def test_dropping_numa_ports_from_a_multi_instance_role_is_rejected(
    tmp_path: Path, live_registry: dict
) -> None:
    """Silent fan-out collapse: consumers read the registry list, not topology."""
    doc = copy.deepcopy(live_registry)
    doc["server_mode"]["frontdoor"].pop("numa_ports")
    doc["server_mode"]["frontdoor"].pop("numa_instances")
    errors = validate_all(_write(tmp_path, doc))
    assert any(
        "frontdoor" in e and "declares no numa_ports" in e for e in errors
    ), errors


def test_primary_port_off_the_full_instance_is_rejected(
    tmp_path: Path, live_registry: dict
) -> None:
    doc = copy.deepcopy(live_registry)
    assert doc["server_mode"]["frontdoor"]["port"] == 8070
    doc["server_mode"]["frontdoor"]["port"] = 8080  # a quarter impersonating the full
    errors = _numa_errors(validate_all(_write(tmp_path, doc)))
    assert any(
        "frontdoor" in e and "full instance port 8070" in e for e in errors
    ), errors


def test_non_integer_numa_ports_are_rejected(tmp_path: Path, live_registry: dict) -> None:
    doc = copy.deepcopy(live_registry)
    doc["server_mode"]["frontdoor"]["numa_ports"] = ["8080", "8180"]
    errors = validate_all(_write(tmp_path, doc))
    assert any("must be a list of ints" in e for e in errors), errors


def test_duplicate_numa_ports_are_rejected(tmp_path: Path, live_registry: dict) -> None:
    doc = copy.deepcopy(live_registry)
    doc["server_mode"]["frontdoor"]["numa_ports"] = [8080, 8080]
    errors = validate_all(_write(tmp_path, doc))
    assert any("duplicate ports" in e for e in errors), errors


def test_validate_or_raise_raises_on_numa_drift(tmp_path: Path, live_registry: dict) -> None:
    """The consumer's call, on the mutated file."""
    doc = copy.deepcopy(live_registry)
    doc["server_mode"]["frontdoor"]["numa_ports"] = [8081, 8181]
    path = _write(tmp_path, doc)
    with pytest.raises(RegistryValidationError, match="disagrees with the NUMA topology"):
        validate_or_raise(path)


# ---------------------------------------------------------------------------
# The check must not pass by having nothing to check
# ---------------------------------------------------------------------------


def test_empty_topology_is_refused_not_treated_as_agreement(live_registry: dict) -> None:
    errors = rv._check_numa_ports_vs_topology(copy.deepcopy(live_registry), numa_config={})
    assert errors and "EMPTY" in errors[0], errors


def test_unloadable_topology_is_reported_not_skipped(monkeypatch, live_registry: dict) -> None:
    def _boom() -> dict:
        raise FileNotFoundError("stack_topology.yaml missing")

    monkeypatch.setattr(rv, "_load_numa_config", _boom)
    errors = rv._check_numa_ports_vs_topology(copy.deepcopy(live_registry))
    assert errors and "could not load the declared topology" in errors[0], errors


def test_registry_without_server_mode_yields_no_false_positives(numa_config: dict) -> None:
    assert rv._check_numa_ports_vs_topology({}, numa_config=numa_config) == []


def test_topology_projection_drops_only_the_full_instance(numa_config: dict) -> None:
    """Pins the projection convention the registry copy is measured against."""
    assert rv._topology_fleet_ports(numa_config["frontdoor"]) == [8080, 8180]
    # 2026-09-27 ARCHITECT SWAP: the single-instance :8074 CPU block is
    # architect_general (Flash-Next); architect_critic is the :8083 GPU host lane.
    assert rv._topology_fleet_ports(numa_config["architect_general"]) == [8074]
    assert rv._topology_fleet_ports(numa_config["architect_critic"]) == [8083]
    assert rv._topology_fleet_ports({"instances": [("0-1", 1234, 2)]}) == [1234]
    assert rv._topology_fleet_ports(
        {"instances": [("0-1", 1, 2), ("2-3", 2, 2)], "full_instance_idx": 0}
    ) == [2]


def test_role_binding_precedence(numa_config: dict) -> None:
    bind = rv._bind_role_to_topology
    # The model_role target is any live topology role other than frontdoor
    # (it was worker_general, whose NUMA_CONFIG entry the 2026-09-22 cutover
    # deleted). `worker` must NOT itself be a topology role, or the model_role
    # branch would never be reached.
    other = next((r for r in sorted(numa_config) if r != "frontdoor"), None)
    assert other is not None, sorted(numa_config)
    assert "worker" not in numa_config
    assert bind("frontdoor", {}, numa_config) == "frontdoor"
    assert bind("worker", {"model_role": other}, numa_config) == other
    assert bind("alias", {"shared_with": ["frontdoor"]}, numa_config) == "frontdoor"
    assert bind("nobody", {"model_role": "not_a_role"}, numa_config) is None
    # direct name outranks model_role, as in stack_manifest.master_server_row
    assert bind("frontdoor", {"model_role": other}, numa_config) == "frontdoor"


# ---------------------------------------------------------------------------
# Wiring — the check must run on the path the launcher takes
# ---------------------------------------------------------------------------


def test_consumer_validates_this_exact_registry() -> None:
    """`stack_commands.cmd_start` gates on `validate_or_raise` over LIVE_REGISTRY."""
    from scripts.server import stack_commands

    source = inspect.getsource(stack_commands.cmd_start)
    assert "from src.registry.registry_validator import validate_or_raise" in source
    assert "validate_or_raise(_registry_yaml)" in source
    assert '"orchestration" / "model_registry.yaml"' in source

    # The path expression the consumer builds, evaluated the same way.
    consumer_registry = (
        Path(stack_commands.__file__).resolve().parent.parent.parent
        / "orchestration"
        / "model_registry.yaml"
    )
    assert consumer_registry == LIVE_REGISTRY


def test_validate_all_actually_calls_the_numa_check(monkeypatch, tmp_path: Path) -> None:
    """Guards the wiring itself: a check nobody calls is not a check.

    Deleting the `errors += _check_numa_ports_vs_topology(registry)` line in
    `validate_all` must fail a test, not just remove coverage.
    """
    calls: list[dict] = []

    def _spy(registry, numa_config=None):  # noqa: ANN001
        calls.append(registry)
        return ["sentinel-numa-error"]

    monkeypatch.setattr(rv, "_check_numa_ports_vs_topology", _spy)
    path = tmp_path / "r.yaml"
    path.write_text("server_mode:\n  frontdoor:\n    port: 8070\n")
    errors = rv.validate_all(path)
    assert calls, "validate_all did not call _check_numa_ports_vs_topology"
    assert "sentinel-numa-error" in errors


# ---------------------------------------------------------------------------
# Pre-existing checks — pinned so the new one cannot be traded against them
# ---------------------------------------------------------------------------


def test_duplicate_yaml_keys_still_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.yaml"
    path.write_text("server_mode:\n  frontdoor:\n    port: 8070\n    port: 8071\n")
    with pytest.raises(RegistryValidationError, match="Duplicate key"):
        validate_all(path)


def test_cross_section_acceleration_conflict_still_rejected(tmp_path: Path) -> None:
    doc = {
        "server_mode": {"frontdoor": {"port": 8070, "acceleration": {"type": "none"}}},
        "roles": {"frontdoor": {"acceleration": {"type": "speculative_decoding"}}},
    }
    errors = validate_all(_write(tmp_path, doc, "acc.yaml"))
    assert any("acceleration.type disagrees" in e for e in errors), errors


def test_same_gguf_two_ports_still_rejected(tmp_path: Path) -> None:
    doc = {
        "server_mode": {
            "a": {"port": 8070, "model": "/models/x.gguf"},
            "b": {"port": 8099, "model": "/models/x.gguf"},
        }
    }
    errors = validate_all(_write(tmp_path, doc, "gguf.yaml"))
    assert any("multiple ports" in e for e in errors), errors


def test_missing_registry_file_still_reported(tmp_path: Path) -> None:
    assert validate_all(tmp_path / "nope.yaml") == [
        f"registry not found: {tmp_path / 'nope.yaml'}"
    ]
