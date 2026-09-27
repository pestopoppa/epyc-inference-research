"""Tests for stack_manifest + stack_paths + stack_commands extraction.

Per the Tranche-7 handoff: "Move constants only after adding tests that import
both `scripts.server.orchestrator_stack` and top-level `orchestrator_stack`."

These tests guard that contract — they verify the registry compiler's fallback
import path (`from orchestrator_stack import ROLE_LAUNCH_META`) keeps working
after the manifest extraction, and that the same names resolve via both
import surfaces.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "server"))


# ----- top-level orchestrator_stack import (registry_compiler fallback path) -----


def test_top_level_orchestrator_stack_import_resolves_role_launch_meta() -> None:
    """src/registry/registry_compiler.py:266 does `from orchestrator_stack import ROLE_LAUNCH_META`
    after `sys.path.insert(0, '/.../scripts/server')`. Must keep working."""
    top_level = importlib.import_module("orchestrator_stack")
    assert hasattr(top_level, "ROLE_LAUNCH_META")
    assert isinstance(top_level.ROLE_LAUNCH_META, dict)
    assert "frontdoor" in top_level.ROLE_LAUNCH_META


def test_top_level_and_package_path_return_same_role_launch_meta() -> None:
    """Both `import orchestrator_stack` and `import scripts.server.orchestrator_stack`
    are valid; they should expose the SAME constant by value."""
    top = importlib.import_module("orchestrator_stack")
    pkg = importlib.import_module("scripts.server.orchestrator_stack")
    assert top.ROLE_LAUNCH_META == pkg.ROLE_LAUNCH_META


# ----- core manifest constants are re-exported from orchestrator_stack -----


@pytest.mark.parametrize(
    "name",
    [
        "PORT_MAP",
        "ROLE_LAUNCH_META",
        "HOT_ROLES",
        "SERIAL_ROLES",
        "NUMA_REPLICA_PORTS",
        "HOT_SERVERS",
        "WARM_SERVERS",
        "EMBEDDING_MODEL_PATH",
        "EMBEDDER_PORTS",
        "WORKER_POOL_MODELS",
        "EXPLORE_DRAFT_MODEL",
        "VISION_WORKER_MODEL",
        "VISION_WORKER_MMPROJ",
        "VISION_ESCALATION_MODEL",
        "VISION_ESCALATION_MMPROJ",
        "DEV_MODEL",
        "DEV_MODEL_PATH",
        "ORCHESTRATOR_PROFILES",
        "DOCKER_SERVICES",
        "validate_model_paths",
        "validate_against_registry",
        "_build_servers_from_classification",
        "_validate_role_classification",
        "_filter_by_numa_mode",
    ],
)
def test_manifest_name_reexported(name: str) -> None:
    stack = importlib.import_module("scripts.server.orchestrator_stack")
    assert hasattr(stack, name), f"{name} not re-exported from orchestrator_stack"


def test_gate3_tool_telemetry_profile_sets_required_api_env() -> None:
    stack = importlib.import_module("scripts.server.orchestrator_stack")
    profile = stack.ORCHESTRATOR_PROFILES["gate3-tool-telemetry"]

    assert profile["AUTOPILOT_TOOL_SENTINELS"] == "1"
    assert profile["ORCHESTRATOR_STRUCTURED_TOOL_OUTPUT"] == "1"


# ----- path/binary constants re-exported from stack_paths via orchestrator_stack -----


@pytest.mark.parametrize(
    "name",
    [
        "_PATHS",
        "STATE_FILE",
        "LLAMA_SERVER",
        "LLAMA_MATH_TOOLS",
        "LLAMA_SERVER_V2",
        "_V2_ROLES",
        "LOG_DIR",
        "SLOT_SAVE_DIR",
        "_get_paths",
        "_HEALTH_SERVER_STARTUP",
        "_HEALTH_VISION_SERVER",
        "_HEALTH_WORKER_SERVER",
    ],
)
def test_path_name_reexported(name: str) -> None:
    stack = importlib.import_module("scripts.server.orchestrator_stack")
    assert hasattr(stack, name), f"{name} not re-exported from orchestrator_stack"


# ----- cmd_* lazy __getattr__ resolution -----


@pytest.mark.parametrize("cmd_name", ["cmd_start", "cmd_stop", "cmd_reload", "cmd_status"])
def test_cmd_name_resolves_lazily_via_getattr(cmd_name: str) -> None:
    """orchestrator_stack.cmd_X resolves via module __getattr__ to stack_commands."""
    stack = importlib.import_module("scripts.server.orchestrator_stack")
    fn = getattr(stack, cmd_name)
    assert callable(fn)


@pytest.mark.parametrize("cmd_name", ["cmd_start", "cmd_stop", "cmd_reload", "cmd_status"])
def test_cmd_name_resolves_to_stack_commands_function(cmd_name: str) -> None:
    stack = importlib.import_module("scripts.server.orchestrator_stack")
    commands = importlib.import_module("scripts.server.stack_commands")

    assert getattr(stack, cmd_name) is getattr(commands, cmd_name)


def test_orchestrator_stack_unknown_attr_raises() -> None:
    """__getattr__ must reject unknown names (not return cmd_* for everything)."""
    stack = importlib.import_module("scripts.server.orchestrator_stack")
    with pytest.raises(AttributeError, match="does_not_exist_xyz"):
        stack.does_not_exist_xyz


# ----- HOT_SERVERS / WARM_SERVERS computed correctly -----


def test_hot_servers_computed_at_module_load() -> None:
    """_build_servers_from_classification is run at stack_manifest module load.
    HOT_SERVERS should be a non-empty list of server dicts."""
    from scripts.server.stack_manifest import HOT_SERVERS

    assert isinstance(HOT_SERVERS, list)
    assert len(HOT_SERVERS) > 0
    for srv in HOT_SERVERS:
        assert "port" in srv
        assert "roles" in srv


def test_hot_servers_includes_frontdoor() -> None:
    from scripts.server.stack_manifest import HOT_SERVERS

    ports = {s["port"] for s in HOT_SERVERS}
    # frontdoor primary on 8070
    assert 8070 in ports


def test_port_map_aliases_match_computed_launch_servers() -> None:
    from scripts.server.stack_manifest import HOT_SERVERS, PORT_MAP, WARM_SERVERS

    computed_role_ports: dict[str, int] = {}
    for server in HOT_SERVERS + WARM_SERVERS:
        for role in server.get("roles", []):
            computed_role_ports.setdefault(role, server["port"])

    for role in ("coder_escalation", "worker_summarize", "worker_math", "toolrunner"):
        assert PORT_MAP[role] == computed_role_ports[role]


def test_dense_retriever_candidates_are_warm_embedding_roles() -> None:
    from scripts.server.stack_manifest import (
        EMBEDDING_SERVER_RECIPES,
        HOT_SERVERS,
        ROLE_LAUNCH_META,
        WARM_SERVERS,
    )

    candidate_ports = {
        "embedder_granite_97m_r2": 8096,
        "embedder_multilingual_e5_base": 8097,
        "embedder_bge_m3": 8098,
    }
    hot_roles = {role for server in HOT_SERVERS for role in server.get("roles", [])}
    warm_role_ports = {
        role: server["port"] for server in WARM_SERVERS for role in server.get("roles", [])
    }

    for role, port in candidate_ports.items():
        assert role not in hot_roles
        assert warm_role_ports[role] == port
        assert ROLE_LAUNCH_META[role]["mode"] == "embedding"
        assert EMBEDDING_SERVER_RECIPES[port]["model_path"].endswith(".gguf")


def test_eval_batch_frontdoor_is_warm_launcher_only() -> None:
    from scripts.server.stack_manifest import HOT_SERVERS, PORT_MAP, ROLE_LAUNCH_META, WARM_SERVERS

    hot_roles = {role for server in HOT_SERVERS for role in server.get("roles", [])}
    warm_servers = {
        role: server for server in WARM_SERVERS for role in server.get("roles", [])
    }

    assert PORT_MAP["eval_batch_frontdoor"] == 18070
    assert "eval_batch_frontdoor" not in hot_roles
    assert warm_servers["eval_batch_frontdoor"]["port"] == 18070
    assert warm_servers["eval_batch_frontdoor"]["eval_batch_frontdoor"] is True
    assert ROLE_LAUNCH_META["eval_batch_frontdoor"]["launcher_only"] is True


def test_launch_kv_quant_configs_keep_canonical_worker_roles_only() -> None:
    """2026-08-02: the table is DERIVED from master, so its keys follow master.

    `worker_explore` used to be absent because the hand-written
    `launch_shape.kv_quant_configs` simply did not list it. That table is deleted;
    the entries now come from `server_mode.<role>.serving_shape.kv_quant` resolved
    over every role name the launcher can be asked about, and worker_explore is a
    declared `shared_with` alias of `worker` — it rides that process and therefore
    genuinely has that process's KV types. Its presence is the derivation working,
    not drift. What the test still pins is that no alias contradicts its host.

    2026-08-04: this test used to end on a hardcoded ABSENCE — "worker_vision is
    not in the table, because it declares no kv_quant while a q8_0 VL quality
    check runs separately". That check has since landed (MMMU-250 paired A/B,
    f16 153/250 vs q8_0 155/250, +0.80 pp, CI [-1.90,+3.03], non-inferior at the
    pre-registered 3 pp margin) and `server_mode.worker_vision.serving_shape.kv_quant`
    is now q8_0/q8_0, so the row exists and MUST exist. Restating a literal
    absence was the defect; every expectation below is now read out of
    `orchestration/model_registry.yaml` directly — the same source of truth the
    resolver reads, resolved independently of it.
    """
    from scripts.server.stack_manifest import LAUNCH_KV_QUANT_CONFIGS

    registry = yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text()
    )["server_mode"]

    def master_kv(launcher_role: str) -> tuple[str, str] | None:
        """Master's KV types for a LAUNCHER role name, resolved from the YAML.

        Deliberately a second implementation of the documented binding order
        (a row named for the role -> a row whose `model_role` is the role -> a
        row whose `shared_with` contains it), grouped `serving_shape` first and
        the flat key second, so the expectation is not produced by the code
        under test.
        """
        row = registry.get(launcher_role)
        if not isinstance(row, dict):
            row = next(
                (
                    cfg
                    for cfg in registry.values()
                    if cfg.get("model_role") == launcher_role
                    or launcher_role in (cfg.get("shared_with") or [])
                ),
                None,
            )
        if not isinstance(row, dict):
            return None
        shape = row.get("serving_shape")
        raw = (shape or {}).get("kv_quant") if isinstance(shape, dict) else None
        if raw is None:
            raw = row.get("kv_quant")
        if not isinstance(raw, dict) or not raw.get("k") or not raw.get("v"):
            return None
        return (str(raw["k"]), str(raw["v"]))

    # Every emitted row equals master's declaration for that role — no launcher
    # row may exist that master does not back.
    assert LAUNCH_KV_QUANT_CONFIGS
    for role, pair in sorted(LAUNCH_KV_QUANT_CONFIGS.items()):
        assert pair == master_kv(role), f"{role} disagrees with master"

    # ...and every role master declares KV types for is present, hosts and
    # aliases alike.
    for role in (
        "frontdoor",
        "worker_general",
        "worker_explore",
        "worker_math",
        "toolrunner",
        "worker_summarize",
        "worker_vision",
        "vision_escalation",
    ):
        expected = master_kv(role)
        assert expected is not None, f"master no longer declares kv_quant for {role}"
        assert LAUNCH_KV_QUANT_CONFIGS[role] == expected

    # No alias may contradict the process it rides — the property this test has
    # always been for.
    for alias in ("worker_explore", "worker_math", "toolrunner"):
        assert LAUNCH_KV_QUANT_CONFIGS[alias] == LAUNCH_KV_QUANT_CONFIGS["worker_general"]
    assert LAUNCH_KV_QUANT_CONFIGS["worker_summarize"] == LAUNCH_KV_QUANT_CONFIGS["frontdoor"]
    assert LAUNCH_KV_QUANT_CONFIGS["vision_escalation"] == LAUNCH_KV_QUANT_CONFIGS["worker_vision"]

    # The "no declaration -> no row -> no -ctk/-ctv, llama-server's f16 default
    # stands" branch, pinned on roles master genuinely says nothing about.
    for undeclared in ("worker_fast", "embedder"):
        assert master_kv(undeclared) is None
        assert undeclared not in LAUNCH_KV_QUANT_CONFIGS


def test_validate_against_registry_checks_port_map_alias_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.server.stack_manifest as manifest

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "process_layout": {
                    "hot_resident": [
                        "frontdoor",
                        "coder_escalation",
                        "worker_summarize",
                        "worker_general",
                        "worker_math",
                        "toolrunner",
                        "architect_general",
                        "ingest_long_context",
                        "worker_vision",
                        "vision_escalation",
                    ],
                    "warm_mmap": [],
                },
                "server_mode": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    drifted = dict(manifest.PORT_MAP)
    drifted["coder_escalation"] = 8071
    monkeypatch.setattr(manifest, "PORT_MAP", drifted)

    warnings = manifest.validate_against_registry(str(registry))

    assert any(
        "role 'coder_escalation': PORT_MAP says port 8071" in warning
        for warning in warnings
    )


def test_validate_against_registry_checks_server_mode_shared_alias_drift(
    tmp_path: Path,
) -> None:
    from scripts.server.stack_manifest import validate_against_registry

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "process_layout": {
                    "hot_resident": [
                        "frontdoor",
                        "coder_escalation",
                        "worker_summarize",
                        "worker_general",
                        "worker_math",
                        "toolrunner",
                        "architect_general",
                        "ingest_long_context",
                        "worker_vision",
                        "vision_escalation",
                    ],
                    "warm_mmap": [],
                },
                "server_mode": {
                    "worker": {
                        "port": 8082,
                        "model_role": "worker_general",
                        "shared_with": ["worker_math", "toolrunner"],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    warnings = validate_against_registry(str(registry))

    assert any(
        "role 'worker': registry server_mode says port 8082 for launch role 'worker_math'"
        in warning
        for warning in warnings
    )


# ----- validate_model_paths is a callable that returns list[str] -----


def test_validate_model_paths_returns_list() -> None:
    from scripts.server.stack_manifest import validate_model_paths

    errors = validate_model_paths()
    assert isinstance(errors, list)
    # all entries should be strings (file paths formatted into error labels)
    for e in errors:
        assert isinstance(e, str)


# ----- numa-mode filter -----


def test_filter_by_numa_mode_both_returns_input_unchanged() -> None:
    from scripts.server.stack_manifest import HOT_SERVERS, _filter_by_numa_mode

    assert _filter_by_numa_mode(HOT_SERVERS, "both") == HOT_SERVERS


def test_filter_by_numa_mode_full_strips_quarters() -> None:
    """For roles with full_instance_idx, mode=full keeps only the full one."""
    from scripts.server.stack_manifest import HOT_SERVERS, _filter_by_numa_mode

    full_only = _filter_by_numa_mode(HOT_SERVERS, "full")
    # frontdoor has 5 instances in NUMA_CONFIG (0=full, 1-4=quarters). With
    # mode=full we keep only the full one — count of frontdoor servers should drop.
    frontdoor_in_full = [s for s in full_only if "frontdoor" in s.get("roles", [])]
    frontdoor_in_all = [s for s in HOT_SERVERS if "frontdoor" in s.get("roles", [])]
    assert len(frontdoor_in_full) < len(frontdoor_in_all)
    assert len(frontdoor_in_full) == 1


# ----- stack_paths is self-contained (no cycle) -----


def test_stack_paths_imports_cleanly() -> None:
    """stack_paths should import without pulling in orchestrator_stack."""
    # Clear cached modules to force a fresh import
    for k in list(sys.modules):
        if k.startswith("scripts.server.stack_paths"):
            del sys.modules[k]
    import scripts.server.stack_paths as sp  # noqa: F401
    # No assertion needed — import success is the test


def test_math_tools_path_supports_subproject_build() -> None:
    """The math-tools binary resolves next to the other kernel binaries.

    Updated 2026-09-22 for the v10 cutover. This used to assert the literal
    substring "build/bin/llama-math-tools", which was the in-tree layout
    (llama.cpp/build/bin). v10 is the first production kernel served from the
    KERNEL STORE, so the path is now
    kernels/builds/<backend>-<date>-<commit>/bin/llama-math-tools and the old
    spelling no longer appears.

    The intent was never the word "build" -- it was that math-tools is found
    beside the kernel it belongs to, rather than at some unrelated path. So the
    assertion now checks THAT: the binary sits in the same directory as
    llama-server. Pinning the store's dated directory name instead would just
    re-break at v11.
    """
    from scripts.server.stack_paths import LLAMA_MATH_TOOLS, LLAMA_SERVER

    assert LLAMA_MATH_TOOLS.name == "llama-math-tools"
    assert LLAMA_MATH_TOOLS.parent == LLAMA_SERVER.parent, (
        f"math-tools ({LLAMA_MATH_TOOLS}) is not beside the server "
        f"({LLAMA_SERVER}); it would be a different kernel build."
    )


# =============================================================================
# Phase 2 — declaration derivation + parity guard
# =============================================================================
# `slots`, `device` and the role alias lists are DERIVED from the master
# registry instead of restated in launch_manifest.yaml. These tests pin both
# halves of that: the derivation produces the declared value, and the guard
# that keeps a second copy from reappearing actually raises.


def test_resolve_slots_returns_the_master_declaration_not_the_serial_policy() -> None:
    """`-np` must come from server_mode.<role>.slots, never from SERIAL_ROLES.

    frontdoor and architect_critic are IN SERIAL_ROLES and DECLARE their slots;
    the old launcher formula (`1 if role in SERIAL_ROLES else 2`) answered the
    admission question with a serving number and returned 1 for both.

    The declared values move (frontdoor 2 -> 16 -> 4, architect_critic 2 -> 4 -> 1
    as master divided every `-np` by 4 on 2026-08-03, operator-ratified), so this
    compares against the DECLARATION rather than a literal — the property under
    test is "resolve_slots returns what master says", which must survive master
    saying something different.

    2026-08-04: the old `slots > 1` line was deleted, not weakened. It was a
    literal proxy for "the SERIAL_ROLES formula did not win", and it cannot
    express that: after the /4 ratification architect_critic legitimately
    declares 1, which is indistinguishable BY VALUE from the old
    `1 if role in SERIAL_ROLES else 2` clamp. The clamp is detectable by SOURCE,
    which is what the `declared` / `source.startswith("master:")` assertions
    below do — the formula could never produce a `master:<role>/...` source. The
    value is checked against a second, independent read of the registry file.
    """
    from scripts.server.stack_manifest import DECLARED_SLOTS, SERIAL_ROLES, resolve_slots

    registry = yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text()
    )["server_mode"]

    # 2026-09-27 ARCHITECT SWAP: the serial full-CPU Flash-Next process is
    # architect_general now (architect_critic is the non-serial :8083 27B).
    for role in ("frontdoor", "architect_general"):
        assert role in SERIAL_ROLES
        decision = resolve_slots(role, "default")
        assert decision.slots == DECLARED_SLOTS[role]
        assert decision.declared
        assert decision.source.startswith("master:")
        master_role = decision.source.split(":", 1)[1].split("/", 1)[0]
        assert registry[master_role]["slots"] == decision.slots


def test_resolve_slots_is_per_instance_for_a_split_role() -> None:
    """One role, two shapes, two answers — what a per-role `slots` cannot express.

    2026-08-04: the 16/4/8 literals were replaced by a second, independent read
    of `serving_shape.slots_by_shape` out of the registry file. They went stale
    when master divided every `-np` by 4 (16 -> 4, 8 -> 2, 4 -> 1) on 2026-08-03;
    the MECHANISM under test — full and half resolve through DIFFERENT shape
    classes, and `declared_slots_by_port` joins that against the topology's ports
    — never moved, and the `.full`/`.half` source suffixes below are what pin it.

    2026-09-23: the ROLE NAMES and PORTS went the same way the slot literals did.
    `worker_general` (8072/8082/8182) stopped being a launch target of its own
    when the CPU fleet host role was renamed and the worker lane became an alias
    set, so the second row of this table named a role with no topology entry and
    a master row reachable only through `shared_with`. WHICH role is split, WHICH
    ports it owns and WHICH master row governs it are now all read out of the
    same three sources the production code reads, so the next lineup move
    re-points this test instead of rotting it.
    """
    from scripts.server.stack_manifest import (
        declared_slots_by_port,
        master_server_row,
        resolve_slots,
    )
    from scripts.server.stack_numa import NUMA_CONFIG, NUMA_INSTANCE_SHAPE_CLASSES

    registry = yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text()
    )["server_mode"]

    def master_row(launcher_role: str) -> tuple[dict, str]:
        """Resolve through the SAME binding master uses: direct, model_role, shared_with.

        Returning the master role NAME as well is what lets the `source` string be
        derived instead of spelled out.
        """
        name, _row, binding = master_server_row(launcher_role)
        assert binding != "unresolved", f"master declares nothing for {launcher_role!r}"
        return registry[name], f"master:{name}/{binding}"

    # WHICH role is split is topology, not a literal: any role whose instances do
    # not all carry the same shape class. Read from stack_numa, the module that
    # owns the table, so a relocated lane is seen here the same way the launcher
    # sees it.
    split_roles = sorted(
        role
        for role, classes in NUMA_INSTANCE_SHAPE_CLASSES.items()
        if len(set(classes)) > 1
    )
    assert split_roles, "the topology declares at least one multi-shape role"

    # The mechanism only has teeth where master declares MORE THAN ONE class for
    # a split role — that is the "two shapes, two answers" case. Assert at least
    # one role reaches it, so this can never pass vacuously through the fallback
    # branch alone.
    multi_class_roles = []

    for role in split_roles:
        classes = NUMA_INSTANCE_SHAPE_CLASSES[role]
        ports = [instance[1] for instance in NUMA_CONFIG[role]["instances"]]
        row, flat_source = master_row(role)
        by_shape = (row.get("serving_shape") or {}).get("slots_by_shape") or {}
        if len(set(classes) & set(by_shape)) > 1:
            multi_class_roles.append(role)

        expected_by_port = {}
        for idx, shape_class in enumerate(classes):
            decision = resolve_slots(role, numa_instance=idx)
            if shape_class in by_shape:
                # (1) the per-instance answer, tagged with the class it came from.
                assert decision.slots == by_shape[shape_class]
                assert decision.source.endswith(f".{shape_class}")
            else:
                # (2) a class master does not declare falls through to the flat
                # compat scalar — documented in `resolve_slots`, and the reason
                # `slots_by_shape` is optional.
                assert decision.slots == row["slots"]
                assert decision.source == flat_source
            expected_by_port[ports[idx]] = decision.slots

        assert declared_slots_by_port(role) == expected_by_port

    assert multi_class_roles, (
        "no split role declares two shape classes in master — the per-instance "
        "branch of resolve_slots would not be exercised at all"
    )

    # A single-shape role has no split and resolves through the flat compat scalar.
    flat_roles = []
    for role, classes in NUMA_INSTANCE_SHAPE_CLASSES.items():
        _name, row, binding = master_server_row(role)
        if row is None or binding == "unresolved":
            continue
        if len(set(classes)) == 1 and "slots_by_shape" not in (
            row.get("serving_shape") or {}
        ):
            flat_roles.append(role)
    assert flat_roles, "at least one role resolves through the flat compat scalar"
    for role in flat_roles:
        row, flat_source = master_row(role)
        flat = resolve_slots(role, numa_instance=0)
        assert (flat.slots, flat.source) == (row["slots"], flat_source)


def test_resolve_slots_matches_master_for_every_declared_role() -> None:
    from scripts.server.stack_manifest import DECLARED_SLOTS, resolve_slots

    registry = yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text()
    )["server_mode"]

    assert DECLARED_SLOTS, "master declares slots for at least the serving roles"
    for role, slots in DECLARED_SLOTS.items():
        decision = resolve_slots(role)
        assert decision.slots == slots
        master_role = decision.source.split(":", 1)[1].split("/", 1)[0]
        assert registry[master_role]["slots"] == slots


def test_resolve_slots_labels_its_fallback_source() -> None:
    """An undeclared role gets the manifest fallback AND says so.

    A fallback that is indistinguishable from a declaration is how "master says
    1 and we launched 2" stayed invisible.
    """
    from scripts.server.stack_manifest import FALLBACK_SLOTS, resolve_slots

    decision = resolve_slots("embedder", "embedding")
    assert not decision.declared
    assert decision.source == "manifest:fallback_slots.embedding"
    assert decision.slots == FALLBACK_SLOTS["embedding"]

    fast = resolve_slots("worker_fast", "worker_pool", worker_type="fast")
    assert fast.source == "manifest:fallback_slots.worker_pool_fast"
    assert fast.slots == 4


def test_vision_device_is_derived_from_master() -> None:
    from scripts.server.stack_manifest import VISION_WORKER_DEVICE

    registry = yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text()
    )["server_mode"]
    assert VISION_WORKER_DEVICE == registry["worker_vision"]["device"]

    manifest = yaml.safe_load(
        (ROOT / "orchestration" / "launch_manifest.yaml").read_text()
    )
    assert "device" not in manifest["vision"]["worker"], (
        "vision.worker must not re-declare a device — it is derived from master"
    )


def test_role_aliases_are_derived_from_master_plus_declared_extras() -> None:
    from scripts.server.stack_manifest import ROLE_LAUNCH_META

    registry = yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text()
    )["server_mode"]
    manifest = yaml.safe_load(
        (ROOT / "orchestration" / "launch_manifest.yaml").read_text()
    )["role_launch_meta"]

    def master_shared(launcher_role: str) -> list[str]:
        row = registry.get(launcher_role)
        if not isinstance(row, dict):
            row = next(
                (cfg for cfg in registry.values() if cfg.get("model_role") == launcher_role),
                None,
            )
        return list((row or {}).get("shared_with") or [])

    for role, meta in ROLE_LAUNCH_META.items():
        extras = list(manifest[role].get("launcher_only_aliases") or [])
        expected = extras + [a for a in master_shared(role) if a not in extras]
        assert meta.get("shared_with_first_n", []) == expected, role
        assert "shared_with_first_n" not in manifest[role], (
            f"{role} re-declares shared_with_first_n; it is derived from master"
        )


def test_parity_guard_passes_on_the_shipped_configuration() -> None:
    from scripts.server.stack_manifest import validate_declaration_parity

    validate_declaration_parity()


@pytest.mark.parametrize(
    "label,target,key,value",
    [
        ("launcher port drifts", "PORT_MAP", "frontdoor", 8071),
        ("launcher tier drifts", "ROLE_LAUNCH_META.frontdoor", "tier", "warm"),
    ],
)
def test_parity_guard_fails_when_a_launcher_row_drifts(
    label: str, target: str, key: str, value: object, monkeypatch
) -> None:
    from scripts.server import stack_manifest as sm

    if "." in target:
        attr, sub = target.split(".", 1)
        container = dict(getattr(sm, attr))
        container[sub] = {**container[sub], key: value}
        monkeypatch.setattr(sm, attr, container)
    else:
        container = dict(getattr(sm, target))
        container[key] = value
        monkeypatch.setattr(sm, target, container)

    with pytest.raises(ValueError, match="parity violated"):
        sm.validate_declaration_parity()


def test_parity_guard_fails_when_master_drifts(monkeypatch) -> None:
    from scripts.server import stack_manifest as sm

    mutated = {
        name: ({**cfg, "port": 9999} if name == "frontdoor" else cfg)
        for name, cfg in sm.MASTER_SERVER_MODE.items()
    }
    monkeypatch.setattr(sm, "MASTER_SERVER_MODE", mutated)

    with pytest.raises(ValueError, match="parity violated"):
        sm.validate_declaration_parity()


def test_parity_guard_fails_when_a_derived_field_is_redeclared(monkeypatch) -> None:
    from scripts.server import stack_manifest as sm

    manifest = {**sm._MANIFEST}
    manifest["vision"] = {
        **manifest["vision"],
        "worker": {**manifest["vision"]["worker"], "device": "ROCm0"},
    }
    monkeypatch.setattr(sm, "_MANIFEST", manifest)

    with pytest.raises(ValueError, match="DERIVED from master"):
        sm.validate_declaration_parity()


def _launcher_role_bound_to_master(sm) -> str:
    """A launcher role that master actually governs — DERIVED, never a literal.

    2026-09-23: the previous caller default here was the literal `worker_general`.
    When the CPU fleet host role was renamed that name left `role_launch_meta`
    entirely, so the synthetic row below landed on NOTHING, step 3 of the guard
    never ran, and both tests went vacuous A SECOND TIME — the exact failure the
    docstring below was written to prevent, minus the assert that would have
    caught it. Derive the role from the same mapping the guard iterates.
    """
    role = next(
        (r for r in sm._MANIFEST["role_launch_meta"] if sm.master_server_row(r)[1] is not None),
        None,
    )
    assert role is not None, "no launcher role resolves to a master server_mode row"
    return role


def _with_launcher_only_alias(sm, monkeypatch, alias: str, *, role: str | None = None) -> str:
    """Give `role` a SYNTHETIC `launcher_only_aliases` row for the duration of a test.

    Returns the role it attached to, so a caller can name the SAME role in its
    `parity.exceptions` row and in its master mutation instead of restating one.

    The two guards below exercise step 3 of `validate_declaration_parity()`,
    which only runs for roles that declare `launcher_only_aliases`. The live
    manifest declares none: the sole real one was deleted on 2026-08-02 when the
    research master started declaring the alias itself, which made the
    launcher-side extra redundant. Both tests used to depend on that production
    row still existing and silently went VACUOUS when it went away. They are
    hermetic now, exactly like the sibling
    `test_numa_prefix_taskset_only_when_no_policy` in tests/unit/test_stack_numa.py:
    the branch stays covered without requiring the shipped config to keep a
    redundant entry alive purely so a unit test has something to assert — and the
    assert below turns "the row landed on nothing" back into a loud failure.
    """
    role = role or _launcher_role_bound_to_master(sm)
    assert role in sm._MANIFEST["role_launch_meta"], (
        f"{role!r} is not a launcher role; the synthetic launcher_only_aliases row "
        f"would land on nothing and step 3 of the guard would never run"
    )
    manifest = {**sm._MANIFEST}
    manifest["role_launch_meta"] = {
        name: ({**meta, "launcher_only_aliases": [alias]} if name == role else meta)
        for name, meta in sm._MANIFEST["role_launch_meta"].items()
    }
    monkeypatch.setattr(sm, "_MANIFEST", manifest)
    assert manifest["role_launch_meta"][role]["launcher_only_aliases"] == [alias]
    return role


def test_parity_guard_fails_on_an_unargued_launcher_only_alias(monkeypatch) -> None:
    """A launcher-only alias with no `parity.exceptions` entry must raise."""
    from scripts.server import stack_manifest as sm

    # A name master declares nowhere — a genuine launcher-only shadow alias.
    assert not any(
        "_fixture_shadow_alias" in (cfg.get("shared_with") or [])
        for cfg in sm.MASTER_SERVER_MODE.values()
    )
    _with_launcher_only_alias(sm, monkeypatch, "_fixture_shadow_alias")
    monkeypatch.setattr(sm, "_PARITY", {**sm._PARITY, "exceptions": []})

    with pytest.raises(ValueError, match="written reason"):
        sm.validate_declaration_parity()


def test_parity_guard_fails_on_a_stale_exception(monkeypatch) -> None:
    """When master starts declaring the alias, the exception must be deleted."""
    from scripts.server import stack_manifest as sm

    # A SYNTHETIC alias, on a DERIVED role, mutated into the master row that role
    # actually binds to. Naming a real alias (`worker_explore`), a real launcher
    # role (`worker_general`) and a real master key (`worker`) pinned this test to
    # one lineup; all three moved on 2026-09-22 and it stopped raising.
    alias = "_fixture_shadow_alias"
    role = _with_launcher_only_alias(sm, monkeypatch, alias)
    master_name, _master_row, _binding = sm.master_server_row(role)
    monkeypatch.setattr(
        sm,
        "_PARITY",
        {
            **sm._PARITY,
            "exceptions": [
                {
                    "field": "shared_with",
                    "launcher_role": role,
                    "reason": "fixture: argued at the time it was written",
                }
            ],
        },
    )
    mutated = {
        name: (
            {**cfg, "shared_with": [alias, *(cfg.get("shared_with") or [])]}
            if name == master_name
            else cfg
        )
        for name, cfg in sm.MASTER_SERVER_MODE.items()
    }
    monkeypatch.setattr(sm, "MASTER_SERVER_MODE", mutated)

    with pytest.raises(ValueError, match="no longer launcher-only"):
        sm.validate_declaration_parity()
