"""Tests for DS-7 Gap 3 + Gap 4 extensions (NIB2-19).

Covers ``ResourceBudget`` dataclass + fine-grained validator checks +
full-restart migration planning (dry-run only — live migration requires
running servers and is not exercised here).
"""

from __future__ import annotations

import pytest

from src.config import stack_templates
from src.config.stack_templates import (
    DEFAULT_MAX_MLOCK_GB,
    DEFAULT_MAX_TOTAL_GB,
    DEFAULT_RESERVE_KV_GB,
    InstanceConfig,
    ResourceBudget,
    RoleConfig,
    StackTemplate,
    load_template,
    validate_template,
)
from src.config.stack_migration import migrate_to_template

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"


def _live_record(ports: list[int]) -> dict:
    return {
        "deployment_status": "live_stack",
        "serving": {
            "endpoint": f"http://localhost:{ports[0]}",
            "ports": ports,
        },
    }


def _make_role(ram_gb: float, tier: str, port: int, n_quarters: int = 0) -> RoleConfig:
    role = RoleConfig(model="dummy", quant="Q4_K_M", tier=tier, ram_gb=ram_gb)
    role.full = InstanceConfig(port=port, numa="NODE0", threads=96)
    for i in range(n_quarters):
        role.quarters.append(
            InstanceConfig(port=port + 100 * (i + 1), numa=f"Q{i}A", threads=48)
        )
    return role


def _make_template(roles: dict[str, RoleConfig], budget: ResourceBudget | None = None) -> StackTemplate:
    t = StackTemplate(name="test", description="", version="1", roles=roles)
    if budget is not None:
        t.resource_budget = budget
    return t


class TestResourceBudgetDefaults:
    def test_defaults_match_system(self):
        b = ResourceBudget()
        assert b.max_mlock_gb == DEFAULT_MAX_MLOCK_GB
        assert b.max_total_gb == DEFAULT_MAX_TOTAL_GB
        assert b.reserve_kv_gb == DEFAULT_RESERVE_KV_GB


class TestValidatorFineGrained:
    def test_hot_budget_exceeded(self):
        roles = {
            "frontdoor": _make_role(500, "HOT", 8000),
            "coder": _make_role(500, "HOT", 8001),
        }
        budget = ResourceBudget(max_mlock_gb=800, max_total_gb=930, reserve_kv_gb=100)
        t = _make_template(roles, budget)
        result = validate_template(t)
        assert not result.valid
        assert any("HOT mlock budget exceeded" in e for e in result.errors)

    def test_total_budget_exceeded(self):
        # 600 HOT + 400 WARM = 1000 > 930 max_total
        roles = {
            "hot": _make_role(600, "HOT", 8000),
            "warm": _make_role(400, "WARM", 8001),
        }
        budget = ResourceBudget(max_mlock_gb=700, max_total_gb=930, reserve_kv_gb=100)
        t = _make_template(roles, budget)
        # Required role check needs frontdoor — inject a trivial one
        t.roles["frontdoor"] = _make_role(1, "HOT", 9000)
        result = validate_template(t)
        assert any("Total loaded budget exceeded" in e for e in result.errors)

    def test_kv_reserve_violation(self):
        # 1000 GB loaded → 130 GB headroom, but reserve_kv_gb=200 → violation
        roles = {"frontdoor": _make_role(1000, "HOT", 8000)}
        budget = ResourceBudget(max_mlock_gb=1100, max_total_gb=1100, reserve_kv_gb=200)
        t = _make_template(roles, budget)
        result = validate_template(t)
        assert any("KV reserve below minimum" in e for e in result.errors)

    def test_warning_when_hot_high(self):
        # 700 HOT vs 800 budget = 87.5% → warning but valid
        roles = {"frontdoor": _make_role(700, "HOT", 8000)}
        budget = ResourceBudget(max_mlock_gb=800, max_total_gb=930, reserve_kv_gb=100)
        t = _make_template(roles, budget)
        result = validate_template(t)
        assert result.valid
        assert any("HOT mlock usage high" in w for w in result.warnings)

    def test_alias_role_requires_existing_target(self):
        roles = {
            "frontdoor": _make_role(10, "HOT", 8000),
            "coder_escalation": RoleConfig(
                model="", quant="", tier="ALIAS", ram_gb=0, alias_to="missing"
            ),
        }
        result = validate_template(_make_template(roles))
        assert not result.valid
        assert any("points to missing target" in e for e in result.errors)

    def test_alias_role_must_not_define_instances(self):
        alias = RoleConfig(
            model="", quant="", tier="ALIAS", ram_gb=0, alias_to="frontdoor"
        )
        alias.full = InstanceConfig(port=8001, numa="NODE0", threads=96)
        roles = {
            "frontdoor": _make_role(10, "HOT", 8000),
            "coder_escalation": alias,
        }
        result = validate_template(_make_template(roles))
        assert not result.valid
        assert any("must not define launch instances" in e for e in result.errors)

    def test_retired_deployable_role_rejected(self):
        roles = {
            "frontdoor": _make_role(10, "HOT", 8000),
            _RETIRED_ARCHITECT_ROLE: _make_role(10, "HOT", 8001),
        }
        result = validate_template(_make_template(roles))
        assert not result.valid
        assert any(f"Retired role '{_RETIRED_ARCHITECT_ROLE}'" in e for e in result.errors)


class TestDefaultYamlRoundTrip:
    def test_default_yaml_loads_and_validates(self):
        t = load_template("default")
        assert t.name == "default"
        assert "frontdoor" in t.roles
        assert t.resource_budget.max_mlock_gb == 800  # from default.yaml
        assert _RETIRED_ARCHITECT_ROLE not in t.roles

        # Alias targets and instance counts are DERIVED from the compiled
        # stack priors — the same artifact `validate_template` checks the
        # template against. default.yaml's own header says these rows must be
        # restated from the compiled artifact and not hand-maintained, so
        # pinning them here would just be a second hand-maintained copy: it is
        # how this test kept asserting `coder_escalation -> frontdoor` after the
        # 2026-08-01 W1 cutover repointed that role at architect_general's
        # :8083 process, and a 5-instance ingest fleet after the 2026-07-30
        # quarters retirement left 1 full + 2 halves.
        records = stack_templates.live_stack_role_records()

        def _prior_ports(role_name: str) -> list[int]:
            return sorted(stack_templates._stack_prior_record_ports(records[role_name]))

        def _expected_alias_target(alias: str) -> str:
            """The deployable template role that serves the alias's prior ports."""
            ports = _prior_ports(alias)
            hosts = [
                name
                for name, role in t.roles.items()
                if not role.alias_to
                and role.tier.upper() != "ALIAS"
                and role.mode != "embedding"
                and sorted(stack_templates._role_instance_ports(role)) == ports
            ]
            assert len(hosts) == 1, (
                f"prior ports {ports} for alias {alias!r} are served by "
                f"{hosts!r} in the template — expected exactly one host role"
            )
            return hosts[0]

        assert t.roles["coder_escalation"].alias_to == _expected_alias_target(
            "coder_escalation"
        )
        assert t.roles["worker_explore"].alias_to == _expected_alias_target(
            "worker_explore"
        )
        assert t.roles["architect_general"].instance_count == len(
            _prior_ports("architect_general")
        )
        # 2026-09-26: ingest_long_context used to be its own fleet and this
        # asserted its instance count. The operator-signed 2026-09-22 cutover
        # (orchestrator 860b0b2d) made it an alias of architect_general's :8083
        # GPU process, so the check is now: it aliases whichever host serves
        # its prior ports, and THAT host realizes every prior port. Same for
        # worker_general, which became an alias of frontdoor's :8070 CPU fleet
        # (the multi-instance case that keeps the count check non-trivial).
        for alias in ("ingest_long_context", "worker_general"):
            host = _expected_alias_target(alias)
            assert t.roles[alias].alias_to == host, alias
            assert t.roles[host].instance_count == len(_prior_ports(alias)), alias
        assert len(_prior_ports("worker_general")) >= 2, "expected a multi-instance host"
        embedder_roles = [
            "embedder",
            "embedder_1",
            "embedder_2",
            "embedder_3",
            "embedder_4",
            "embedder_5",
        ]
        assert [t.roles[name].full.port for name in embedder_roles] == [
            8090,
            8091,
            8092,
            8093,
            8094,
            8095,
        ]
        assert {t.roles[name].mode for name in embedder_roles} == {"embedding"}
        assert {t.roles[name].full.threads for name in embedder_roles} == {4}
        result = validate_template(t)
        assert result.valid, f"default template should validate: {result.errors}"

    def test_default_yaml_rejects_generated_prior_port_drift(self, monkeypatch):
        t = load_template("default")
        live_records = {
            role_name: _live_record(stack_templates._role_instance_ports(role))
            for role_name, role in t.roles.items()
            if not role.alias_to
            and role.tier.upper() != "ALIAS"
            and role.mode != "embedding"
        }
        live_records["frontdoor"] = _live_record([8070, 8080, 8180, 8280, 8399])
        monkeypatch.setattr(
            stack_templates,
            "live_stack_role_records",
            lambda: live_records,
        )

        result = validate_template(t)

        assert not result.valid
        assert any("frontdoor" in error and "generated stack-prior ports" in error for error in result.errors)

    def test_default_yaml_allows_generated_alias_ports_on_alias_target(self, monkeypatch):
        """An alias's generated ports validate when its TARGET serves them.

        The synthetic alias ports are taken from the alias's own declared target
        instead of being restated as literals. The old literals (coder_escalation
        -> 8070, worker_math -> 8072/8082) encoded a target that has since moved:
        the 2026-08-01 W1 cutover repointed coder_escalation from frontdoor to
        architect_general's :8083. Pinning 8070 stopped testing "served by the
        target" and started testing "served by frontdoor specifically", which is
        a different — and now false — claim.
        """
        t = load_template("default")
        live_records = {
            role_name: _live_record(stack_templates._role_instance_ports(role))
            for role_name, role in t.roles.items()
            if not role.alias_to
            and role.tier.upper() != "ALIAS"
            and role.mode != "embedding"
        }

        # Two shapes an alias's generated record can take: the target's FULL port
        # list, and a strict subset of it. Both must validate.
        alias_names = [
            name
            for name, role in t.roles.items()
            if role.alias_to and role.mode != "embedding"
        ]
        assert alias_names, "default template must declare alias roles"
        checked_full = checked_subset = None
        for name in sorted(alias_names):
            target_ports = stack_templates._role_instance_ports(
                t.roles[t.roles[name].alias_to]
            )
            assert target_ports, f"alias target for {name} serves no ports"
            if len(target_ports) > 1 and checked_subset is None:
                live_records[name] = _live_record(target_ports[:-1])
                checked_subset = name
            else:
                live_records[name] = _live_record(target_ports)
                if checked_full is None:
                    checked_full = name
        assert checked_full and checked_subset, (
            "both the full-port and subset alias shapes must be exercised"
        )
        monkeypatch.setattr(
            stack_templates,
            "live_stack_role_records",
            lambda: live_records,
        )

        result = validate_template(t)

        assert result.valid, result.errors

        # Negative control: a port the target does NOT serve must still be
        # rejected, so the assertion above cannot pass by accepting anything.
        target_ports = stack_templates._role_instance_ports(
            t.roles[t.roles[checked_full].alias_to]
        )
        stray = max(target_ports) + 1000
        assert stray not in target_ports
        live_records[checked_full] = _live_record([*target_ports, stray])
        rejected = validate_template(t)
        assert not rejected.valid
        assert any(
            checked_full in error and "not served by target ports" in error
            for error in rejected.errors
        ), rejected.errors

    def test_non_default_template_does_not_require_stack_prior_port_parity(self, monkeypatch):
        t = StackTemplate(
            name="experimental",
            roles={"frontdoor": _make_role(10, "HOT", 9999)},
        )
        monkeypatch.setattr(
            stack_templates,
            "live_stack_role_records",
            lambda: {"frontdoor": _live_record([8070])},
        )

        result = validate_template(t)

        assert result.valid, result.errors

    def test_default_yaml_preserves_ds7_decision_metadata(self):
        t = load_template("default")
        assert t.metadata["ds7_profile"] == "steady_state_static_prewarm"
        decision = t.metadata["ds7_decision"]
        assert decision["status"] == "retain_default"
        assert decision["ds6_quarter_scheduler"] == "parked_until_static_prewarm_gap"
        assert decision["evidence_packet"].endswith(
            "ds_e1_evidence_packet_20260704T192333Z.md"
        )

    def test_hot_vs_loaded_breakdown(self):
        t = load_template("default")
        # The default stack is no longer all-HOT: STACKCHG-DFLASH2-20261003 made worker_vision
        # WARM (cold CPU, not started by a default start). loaded = hot + every WARM role,
        # recomputed from the template rather than pinned. (This failed on HEAD since d3233170.)
        warm = {name: r for name, r in t.roles.items() if r.tier == "WARM"}
        assert "worker_vision" in warm
        assert t.loaded_ram_gb == pytest.approx(
            t.hot_ram_gb + sum(r.total_ram_gb for r in warm.values()))
        assert t.hot_ram_gb > 0


class TestMigrationDryRun:
    def test_dry_run_default_noop(self):
        result = migrate_to_template("default", dry_run=True)
        assert result.ok
        assert result.dry_run
        phase_names = [p.name for p in result.phases]
        assert phase_names == ["save_kv", "stop_all", "start_target", "restore_kv", "verify_health"]
        # In dry-run, stop/start/restore/verify are skipped
        skipped = {p.name for p in result.phases if p.status == "skipped"}
        assert "stop_all" in skipped
        assert "start_target" in skipped

    def test_missing_template_fails(self):
        result = migrate_to_template("this-template-does-not-exist", dry_run=True)
        assert not result.ok
        assert "not found" in result.reason.lower()

    def test_summary_is_multiline(self):
        result = migrate_to_template("default", dry_run=True)
        s = result.summary()
        assert "DRY-RUN" in s
        assert "save_kv" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
