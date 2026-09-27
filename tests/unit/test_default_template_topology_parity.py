"""P1-4: pin `stack_templates/default.yaml`'s topology to the single source.

The template is the FOURTH hand-maintained copy of "which ports, on which
cpuset, with how many threads". The single source is
`orchestration/stack_topology.yaml` `numa_config:`, exposed as
``scripts.server.stack_numa.NUMA_CONFIG``.

Every test here perturbs ONE side and asserts the gate goes red. A parity test
that only ever runs against agreeing inputs proves nothing: it cannot tell
"they match" from "I compared nothing". So each invariant is asserted in both
directions -- green on the real pair, red on a mutant -- and the green case
additionally asserts that the compared set was non-empty and plausibly sized.

All tests are module/class-level pytest functions. Assertions inside a
``main()`` are NOT counted by the reporter; that exact defect shipped in this
repo, so verify with ``--collect-only`` when editing this file.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.validate.default_template_topology_parity import (
    DEFAULT_TEMPLATE_PATH,
    MIN_COMPARED_INSTANCES,
    MIN_COMPARED_ROLES,
    SOURCE_ROLES_NOT_IN_TEMPLATE,
    Instance,
    ParityReport,
    check_parity,
    derive_expected,
    load_source,
    load_template_document,
    run_check,
    shape_to_template_numa,
    template_numa_to_shape,
)

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml"


def _registry_host_fleets() -> frozenset[str]:
    """Registry `server_mode` rows that own a llama-server process of their own.

    A row is a host unless it declares `alias_of` (it rides another row's
    process). This is a source INDEPENDENT of numa_config, so asserting every
    host is compared is not a restatement of what the gate iterates.
    """
    doc = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    server_mode = doc["server_mode"]
    return frozenset(
        role
        for role, row in server_mode.items()
        if isinstance(row, dict) and not row.get("alias_of")
    )


# Roles that must be covered. A floor, not the answer: if the fleet grows, the
# role-set accounting in `check_parity` fails on the new role, not this list.
#
# 2026-09-26: this used to be a literal naming worker_general and
# ingest_long_context as fleets. The operator-signed 2026-09-22 cutover
# (orchestrator 860b0b2d) made them aliases (of frontdoor's :8070 CPU fleet and
# architect_general's :8083 GPU process — architect_critic since the 2026-09-27
# swap) and deleted their numa_config entries.
# The set is now DERIVED from the registry's host rows, so the next lineup
# change moves it with the registry.
REQUIRED_COVERED_ROLES = _registry_host_fleets()
assert "frontdoor" in REQUIRED_COVERED_ROLES and len(REQUIRED_COVERED_ROLES) >= 3, (
    f"registry host-fleet derivation looks vacuous: {sorted(REQUIRED_COVERED_ROLES)}"
)


def _multi_instance_role(source) -> str:
    """A sourced (non-excepted) role with a full + sibling fleet, derived from
    numa_config -- the mutation target the literal worker_general /
    ingest_long_context used to be."""
    numa_config = source[0]
    multi = sorted(
        role
        for role, cfg in numa_config.items()
        if len(cfg.get("instances") or []) > 1 and role not in SOURCE_ROLES_NOT_IN_TEMPLATE
    )
    assert multi, f"no multi-instance sourced role; roles={sorted(numa_config)}"
    return multi[0]


@pytest.fixture(scope="module")
def source():
    numa_config, instance_shapes, known_shapes = load_source()
    return copy.deepcopy(numa_config), copy.deepcopy(instance_shapes), dict(known_shapes)


@pytest.fixture(scope="module")
def template_document():
    return load_template_document()


def _check(source, document, **kwargs) -> ParityReport:
    numa_config, instance_shapes, known_shapes = source
    return check_parity(numa_config, instance_shapes, document, known_shapes, **kwargs)


def _mutate_source(source, role, index, *, port=None, threads=None, shape=None):
    """Return a copy of `source` with one instance field changed."""
    numa_config, instance_shapes, known_shapes = copy.deepcopy(source[0]), copy.deepcopy(
        source[1]
    ), dict(source[2])
    instances = list(numa_config[role]["instances"])
    cpuset, cur_port, cur_threads = instances[index]
    instances[index] = (
        cpuset,
        cur_port if port is None else port,
        cur_threads if threads is None else threads,
    )
    numa_config[role] = dict(numa_config[role])
    numa_config[role]["instances"] = instances
    if shape is not None:
        shapes = list(instance_shapes[role])
        shapes[index] = shape
        instance_shapes[role] = tuple(shapes)
    return numa_config, instance_shapes, known_shapes


def _mutate_template(document, mutate):
    doc = copy.deepcopy(document)
    mutate(doc["roles"])
    return doc


# ---------------------------------------------------------------------------
# green case + non-vacuity
# ---------------------------------------------------------------------------


class TestGateIsGreenAndNotVacuous:
    def test_live_default_template_matches_the_single_source(self):
        report = run_check()
        assert report.ok, "default.yaml topology has drifted from " f"stack_topology.yaml numa_config:\n  " + "\n  ".join(
            report.problems
        )

    def test_comparison_set_is_non_empty(self, source, template_document):
        """Guard the vacuous pass: prove the gate compared something."""
        report = _check(source, template_document)
        assert report.compared_roles, "no roles were compared at all"
        assert report.compared_instances > 0, "no instances were compared at all"

    def test_comparison_set_is_plausibly_sized(self, source, template_document):
        report = _check(source, template_document)
        assert len(report.compared_roles) >= MIN_COMPARED_ROLES, (
            f"compared only {report.compared_roles!r}"
        )
        assert report.compared_instances >= MIN_COMPARED_INSTANCES, (
            f"compared only {report.compared_instances} instances"
        )

    def test_every_required_role_is_actually_covered(self, source, template_document):
        """Naming the roles, so a role silently dropping out is a failure."""
        report = _check(source, template_document)
        missing = REQUIRED_COVERED_ROLES - set(report.compared_roles)
        assert not missing, f"these roles were not compared by the gate: {sorted(missing)}"

    def test_template_file_is_the_one_on_disk(self):
        assert DEFAULT_TEMPLATE_PATH.exists()
        assert DEFAULT_TEMPLATE_PATH.name == "default.yaml"


# ---------------------------------------------------------------------------
# mutation: perturb the TEMPLATE (the copy)
# ---------------------------------------------------------------------------


class TestTemplateSideMutationsGoRed:
    def test_changed_port_fails(self, source, template_document):
        role = _multi_instance_role(source)
        doc = _mutate_template(
            template_document, lambda r: r[role]["full"].update(port=9999)
        )
        report = _check(source, doc)
        assert not report.ok
        assert any("9999" in p for p in report.problems)

    def test_changed_numa_shape_fails(self, source, template_document):
        doc = _mutate_template(
            template_document, lambda r: r["worker_vision"]["full"].update(numa="FULL")
        )
        report = _check(source, doc)
        assert not report.ok
        assert any("numa shape" in p and "worker_vision" in p for p in report.problems)

    def test_changed_threads_fails(self, source, template_document):
        """This is defect c3: an SMT-oversubscribed thread count frozen into a
        template that no gate re-derives. It must go red now.

        The mutated row is the 8-thread GPU host-lane 27B (:8083) — architect_critic
        since the 2026-09-27 ARCHITECT SWAP. (architect_general is now the 96-thread
        full-CPU instance, where threads=96 is the TRUE value and would not go red.)"""
        doc = _mutate_template(
            template_document, lambda r: r["architect_critic"]["full"].update(threads=96)
        )
        report = _check(source, doc)
        assert not report.ok
        assert any("threads is 96" in p for p in report.problems)

    def test_omitted_threads_key_fails(self, source, template_document):
        """Omitting the field must NOT pass. `load_template` would substitute a
        default here; reading the raw YAML is what makes the omission visible."""

        def drop(roles):
            roles["frontdoor"]["full"].pop("threads")

        report = _check(source, _mutate_template(template_document, drop))
        assert not report.ok
        assert any("threads is 0" in p for p in report.problems)

    def test_omitted_numa_key_fails(self, source, template_document):
        def drop(roles):
            roles["frontdoor"]["full"].pop("numa")

        report = _check(source, _mutate_template(template_document, drop))
        assert not report.ok
        assert any("numa shape is ''" in p for p in report.problems)

    def test_dropped_sibling_instance_fails(self, source, template_document):
        """A retired-lineup edit that removes a half must be caught."""

        role = _multi_instance_role(source)
        assert len(template_document["roles"][role]["quarters"]) >= 2, role

        def drop(roles):
            roles[role]["quarters"] = roles[role]["quarters"][:1]

        report = _check(source, _mutate_template(template_document, drop))
        assert not report.ok
        assert any("missing source port" in p for p in report.problems)

    def test_revived_retired_quarter_port_fails(self, source, template_document):
        """The four-quarter lineup was retired and its ports freed. Reviving one
        in the template is the exact regression the header forbids."""

        def revive(roles):
            roles["frontdoor"]["quarters"].append(
                {"port": 8280, "numa": "Q1A", "threads": 48}
            )

        report = _check(source, _mutate_template(template_document, revive))
        assert not report.ok
        assert any("8280" in p and "source does not" in p for p in report.problems)

    def test_role_deleted_from_template_fails(self, source, template_document):
        def delete(roles):
            del roles["architect_critic"]

        report = _check(source, _mutate_template(template_document, delete))
        assert not report.ok
        assert any(
            "architect_critic" in p and "absent from the template" in p
            for p in report.problems
        )

    def test_alias_that_declares_instances_fails(self, source, template_document):
        def add(roles):
            roles["worker_math"]["full"] = {"port": 8099, "numa": "FULL", "threads": 96}

        report = _check(source, _mutate_template(template_document, add))
        assert not report.ok
        assert any("is an alias but declares" in p for p in report.problems)

    def test_unsourced_deployable_role_fails(self, source, template_document):
        """A brand-new hand-written topology row with no numa_config entry is
        a fifth copy. It must not be silently skipped."""

        def add(roles):
            roles["ghost_role"] = {
                "model": "x",
                "quant": "Q4_K_M",
                "tier": "HOT",
                "ram_gb": 1,
                "full": {"port": 8199, "numa": "FULL", "threads": 96},
            }

        report = _check(source, _mutate_template(template_document, add))
        assert not report.ok
        assert any("unsourced topology copy" in p for p in report.problems)

    def test_embedder_given_numa_placement_fails(self, source, template_document):
        def place(roles):
            roles["embedder_3"]["full"]["numa"] = "HALF_B"

        report = _check(source, _mutate_template(template_document, place))
        assert not report.ok
        assert any("embedding-mode but declares NUMA" in p for p in report.problems)

    def test_empty_template_fails_loudly_instead_of_passing_vacuously(self, source):
        report = _check(source, {"roles": {}})
        assert not report.ok
        assert any(p.startswith("VACUOUS:") for p in report.problems)

    def test_template_with_no_roles_key_fails_loudly(self, source):
        report = _check(source, {})
        assert not report.ok
        assert any(p.startswith("VACUOUS:") for p in report.problems)


# ---------------------------------------------------------------------------
# mutation: perturb the SOURCE
# ---------------------------------------------------------------------------


class TestSourceSideMutationsGoRed:
    def test_source_port_change_fails(self, source, template_document):
        mutated = _mutate_source(source, "frontdoor", 0, port=8071)
        report = _check(mutated, template_document)
        assert not report.ok
        assert any("8071" in p for p in report.problems)

    def test_source_threads_change_fails(self, source, template_document):
        role = _multi_instance_role(source)
        original = source[0][role]["instances"][1][2]
        assert original != 24, "mutation must actually change the thread count"
        mutated = _mutate_source(source, role, 1, threads=24)
        report = _check(mutated, template_document)
        assert not report.ok
        assert any(
            f"threads is {original} in the template but 24" in p for p in report.problems
        )

    def test_source_shape_change_fails(self, source, template_document):
        mutated = _mutate_source(source, "frontdoor", 1, shape="NUMA_Q0A")
        report = _check(mutated, template_document)
        assert not report.ok
        assert any("numa shape" in p and "Q0A" in p for p in report.problems)

    def test_new_source_role_not_in_template_fails(self, source, template_document):
        numa_config, instance_shapes, known_shapes = (
            copy.deepcopy(source[0]),
            copy.deepcopy(source[1]),
            dict(source[2]),
        )
        numa_config["brand_new_role"] = {
            "instances": [("0-95", 8199, 96)],
            "full_instance_idx": 0,
        }
        instance_shapes["brand_new_role"] = ("NUMA_FULL",)
        report = _check((numa_config, instance_shapes, known_shapes), template_document)
        assert not report.ok
        assert any(
            "brand_new_role" in p and "declared exception" in p for p in report.problems
        )

    def test_empty_source_fails_loudly_instead_of_passing_vacuously(
        self, source, template_document
    ):
        """`test_stack_change_guard.py` writes a `stack_numa.py` with
        `NUMA_CONFIG = {}`. If that shape ever reached this gate it must fail,
        not report zero mismatches over zero roles."""
        report = _check(({}, {}, source[2]), template_document)
        assert not report.ok
        assert any(p.startswith("VACUOUS:") for p in report.problems)

    def test_source_with_mismatched_shape_count_fails(self, source, template_document):
        numa_config, instance_shapes, known_shapes = (
            copy.deepcopy(source[0]),
            copy.deepcopy(source[1]),
            dict(source[2]),
        )
        instance_shapes["frontdoor"] = instance_shapes["frontdoor"][:1]
        report = _check((numa_config, instance_shapes, known_shapes), template_document)
        assert not report.ok
        assert any("internally inconsistent" in p for p in report.problems)


# ---------------------------------------------------------------------------
# the exception list is an exception list, not a skip list
# ---------------------------------------------------------------------------


class TestExceptionListIsAccountable:
    def test_every_declared_exception_carries_a_reason(self):
        assert SOURCE_ROLES_NOT_IN_TEMPLATE, "the exception list must be explicit"
        for role, reason in SOURCE_ROLES_NOT_IN_TEMPLATE.items():
            assert isinstance(reason, str) and len(reason) > 20, (
                f"exception {role!r} has no substantive reason"
            )

    def test_every_declared_exception_is_really_in_the_source(self, source):
        numa_config = source[0]
        for role in SOURCE_ROLES_NOT_IN_TEMPLATE:
            assert role in numa_config, (
                f"{role!r} is excepted but no longer exists in numa_config -- "
                "the exception is stale and would hide a future role of that name"
            )

    def test_clearing_the_exception_list_turns_the_gate_red(
        self, source, template_document
    ):
        """Proves the exception is load-bearing: without it, the omission is a
        failure. An exception list nothing depends on is a skip list."""
        report = _check(source, template_document, source_roles_not_in_template={})
        assert not report.ok
        assert any(
            "eval_batch_frontdoor" in p and "absent from the template" in p
            for p in report.problems
        )


# ---------------------------------------------------------------------------
# derivation helpers
# ---------------------------------------------------------------------------


class TestDerivationHelpers:
    @pytest.mark.parametrize(
        ("shape", "expected"),
        [
            ("NUMA_FULL", "FULL"),
            ("NUMA_HALF_A", "HALF_A"),
            ("NUMA_HALF_B", "HALF_B"),
            ("GPU_HOST_LANE", "GPU_HOST_LANE"),
        ],
    )
    def test_shape_name_round_trips(self, shape, expected, source):
        known = source[2]
        assert shape_to_template_numa(shape) == expected
        assert template_numa_to_shape(expected, known) == shape

    def test_derived_instances_agree_with_cpu_shapes(self, source):
        """The derived `numa:` name must resolve to the cpuset actually pinned --
        otherwise the template names a shape the launcher cannot produce."""
        numa_config, instance_shapes, known_shapes = source
        expected = derive_expected(numa_config, instance_shapes)
        checked = 0
        for role, cfg in numa_config.items():
            for (cpuset, port, threads), inst in zip(
                cfg["instances"], expected[role]["instances"]
            ):
                assert isinstance(inst, Instance)
                shape_key = template_numa_to_shape(inst.numa, known_shapes)
                assert known_shapes[shape_key] == (cpuset, threads), (
                    f"{role}:{port} derived shape {inst.numa!r} -> {shape_key!r} "
                    f"is {known_shapes[shape_key]}, not ({cpuset!r}, {threads})"
                )
                checked += 1
        assert checked >= MIN_COMPARED_INSTANCES, f"only checked {checked} instances"

    def test_full_port_is_the_sources_full_instance(self, source):
        numa_config, instance_shapes, _known = source
        expected = derive_expected(numa_config, instance_shapes)
        multi = [r for r, e in expected.items() if len(e["instances"]) > 1]
        assert multi, "expected at least one multi-instance role"
        for role in multi:
            idx = numa_config[role]["full_instance_idx"]
            assert expected[role]["full_port"] == numa_config[role]["instances"][idx][1]


# ---------------------------------------------------------------------------
# the CLI is the evidence command; it must agree with the library
# ---------------------------------------------------------------------------


class TestCli:
    def test_cli_exit_code_matches_report(self, tmp_path: Path, template_document):
        from scripts.validate import default_template_topology_parity as mod

        assert mod.main(["--template", str(DEFAULT_TEMPLATE_PATH)]) == 0

        drifted = copy.deepcopy(template_document)
        drifted["roles"]["frontdoor"]["full"]["threads"] = 48
        path = tmp_path / "drifted.yaml"
        path.write_text(yaml.safe_dump(drifted), encoding="utf-8")
        assert mod.main(["--template", str(path)]) == 1

    def test_render_emits_every_sourced_role(self):
        from scripts.validate import default_template_topology_parity as mod

        rendered = mod.render_derived_blocks()
        for role in REQUIRED_COVERED_ROLES:
            assert f"  {role}:" in rendered
        for role in SOURCE_ROLES_NOT_IN_TEMPLATE:
            assert f"  {role}:" not in rendered
        assert yaml.safe_load(rendered), "rendered blocks must be valid YAML"
