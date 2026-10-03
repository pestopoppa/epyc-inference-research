"""Registry consistency validator (added 2026-05-09).

Catches the failure modes that cost ~2 hours of debugging on 2026-05-09:

1. **Cross-section conflicts**: a role appears in both `server_mode.roles.X`
   and top-level `roles.X` with disagreeing values for fields the launcher
   reads from one section but the runtime reads from the other (e.g.
   `acceleration.type`). Edits to one path silently no-op.

2. **GGUF/port inconsistency**: two roles share the same `model.full_path`
   (= same llama-server should host them) but declare different `port:`
   values. This was the frontdoor + coder_escalation duplicate that wasted
   36 GB of mlocked weights and ran two competing OMP teams.

3. **YAML duplicate keys at the same path**: PyYAML silently keeps the LAST
   declaration. Detect and reject.

5. **Drafter selection** (DRAFT-SEL-1, 2026-10-01): the drafter a server runs must
   be one its model's master `drafters` list declares, chosen by
   `stack_topology.yaml -> drafter_selection`; see `_check_drafter_selection`.

4. **`numa_ports` / `numa_instances` disagreeing with the declared NUMA
   topology** (added 2026-08-12, handoff P1-5). `server_mode.<role>.numa_ports`
   is the registry's copy of a fleet that `orchestration/stack_topology.yaml`
   (loaded as `scripts.server.stack_numa.NUMA_CONFIG`) owns. Two copies of one
   fleet is how months of NUMA drift went unnoticed: the registry-side copy is
   what `src/registry/model_descriptors.py`, `src/registry/stack_priors.py` and
   `scripts/autopilot/eval_tower.py` turn into backend URLs, so a stale copy
   points eval fan-out and capacity planning at ports nothing is listening on
   — silently, because every consumer just reads the list it was given.

   `scripts/server/stack_manifest.py:validate_declaration_parity()` already
   compares the two for roles the topology KNOWS (it iterates `NUMA_CONFIG`).
   It structurally cannot see the other direction: a registry role that
   declares a fleet the topology has never heard of is skipped by that loop
   entirely. That is the exact shape of the `vision_escalation` phantom fleet
   (ports 8187/8287/8387/8487, declared for a role with no `NUMA_CONFIG`
   entry). This check covers both directions on the file the launcher actually
   loads.

Wire-in (intended): orchestrator_stack.py:cmd_start calls `validate_all` on
the resolved registry path before any RegistryLoader instance, fails fast
on errors.

Wire-in (actual): `scripts/server/stack_commands.py:cmd_start` calls
`validate_or_raise(orchestration/model_registry.yaml)` — the COMPILED LEAN
registry, i.e. the artifact the orchestrator consumes — and returns 2 without
starting anything if it raises.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


# Fields that, if declared in BOTH `server_mode.X` and `roles.X`,
# must agree across both (or only be declared in one).
# NOTE: `model` and `draft_model` are intentionally NOT here — server_mode
# uses short filename strings while roles uses structured dicts. They're
# different representations by design; comparing them produces noise.
# The model-path consistency check happens separately in
# `_check_gguf_port_consistency` which compares resolved GGUF paths.
_DUAL_DECLARED_CONFLICT_FIELDS = (
    "acceleration",
)


class RegistryValidationError(Exception):
    """Registry violates a consistency invariant."""


def _strict_yaml_load(path: Path) -> dict[str, Any]:
    """Load YAML and reject duplicate keys at the same mapping level.

    PyYAML's default safe_load silently keeps the last value for a duplicate
    key. We override the constructor to raise instead.
    """

    class _StrictLoader(yaml.SafeLoader):
        pass

    def _no_duplicate_keys(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise RegistryValidationError(
                    f"Duplicate key {key!r} at line {key_node.start_mark.line + 1} "
                    f"in {path} (first defined earlier in same mapping)"
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _StrictLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys
    )

    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_StrictLoader)


def _check_cross_section_conflicts(registry: dict[str, Any]) -> list[str]:
    """Check that role-level fields don't disagree between server_mode and roles sections.

    The registry layout is:
        server_mode:
          <role_name>:    # launch config: port, slots, model, draft_model, acceleration, ...
          ...
        roles:
          <role_name>:    # model metadata: tier, description, model, acceleration, ...
          ...

    Both sections may declare overlapping fields. They MUST agree (or the field
    must only live in one section).
    """
    errors: list[str] = []
    server_mode = registry.get("server_mode") or {}
    top_roles = registry.get("roles") or {}

    # server_mode keys are role names directly (not nested under 'roles').
    # Filter to dict values only (other server_mode keys may be scalar config).
    server_roles = {k: v for k, v in server_mode.items() if isinstance(v, dict) and k in top_roles}

    for role in sorted(server_roles):
        s = server_roles[role]
        t = top_roles.get(role) or {}
        for field in _DUAL_DECLARED_CONFLICT_FIELDS:
            if field in s and field in t and s[field] != t[field]:
                # For acceleration we compare type (the most-load-bearing key);
                # avoids spurious conflicts on benchmark-date or notes fields.
                if field == "acceleration":
                    s_type = s[field].get("type") if isinstance(s[field], dict) else None
                    t_type = t[field].get("type") if isinstance(t[field], dict) else None
                    if s_type == t_type:
                        continue
                    errors.append(
                        f"role {role!r}: acceleration.type disagrees between "
                        f"server_mode.{role}.acceleration ({s_type!r}) and "
                        f"roles.{role}.acceleration ({t_type!r}). "
                        f"Pick one source of truth; remove the other section's "
                        f"acceleration block. Today's RegistryLoader reads roles.X."
                    )
                else:
                    errors.append(
                        f"role {role!r}: {field!r} disagrees between sections. "
                        f"server_mode.{role}.{field} = {s[field]!r:.80}; "
                        f"roles.{role}.{field} = {t[field]!r:.80}. "
                        f"Pick one section as authoritative."
                    )
    return errors


def _check_gguf_port_consistency(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    by_model: dict[str, list[tuple[str, int]]] = defaultdict(list)

    # server_mode keys are role names directly. Filter to dict values.
    server_mode = registry.get("server_mode") or {}
    top_roles = registry.get("roles") or {}
    server_roles = {k: v for k, v in server_mode.items() if isinstance(v, dict)}

    for role, cfg in server_roles.items():
        if not isinstance(cfg, dict):
            continue
        port = cfg.get("port")
        if port is None:
            continue
        # Look up the model path: prefer server_mode.roles.X.model_path,
        # fall back to top-level roles.X.model.path or .full_path.
        model_path = cfg.get("model_path") or cfg.get("model")
        if not model_path or not isinstance(model_path, str):
            top = top_roles.get(role) or {}
            model_field = top.get("model")
            if isinstance(model_field, dict):
                model_path = model_field.get("path") or model_field.get("full_path")
        if model_path and isinstance(model_path, str):
            by_model[model_path].append((role, int(port)))

    for model_path, role_ports in by_model.items():
        ports = {p for _, p in role_ports}
        if len(ports) > 1:
            roles_list = ", ".join(f"{r}:{p}" for r, p in sorted(role_ports))
            errors.append(
                f"GGUF {model_path.split('/')[-1]!r} is declared with "
                f"multiple ports ({sorted(ports)}). Roles: {roles_list}. "
                f"Same-GGUF roles must share one llama-server (same port); "
                f"set one canonical port for the whole group."
            )
    return errors


def _load_numa_config() -> dict[str, Any]:
    """The declared NUMA topology, as the launcher sees it.

    Deliberately NOT a second YAML reader over `stack_topology.yaml`: this
    imports the same `NUMA_CONFIG` object every other consumer imports, so a
    change to how the topology is loaded (cpu-shape expansion, field
    validation, the `full_instance_idx` convention) cannot drift away from this
    guard. A local re-parse would be a fourth copy of the topology, which is
    the defect class this check exists to catch.

    Raises on failure. There is no fallback table and no `except: return {}` —
    a cross-check that silently degrades to "no topology, therefore no
    disagreement" passes hardest exactly when the topology is broken.
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from scripts.server.stack_numa import NUMA_CONFIG  # noqa: PLC0415

    return NUMA_CONFIG


def _topology_fleet_ports(cfg: dict[str, Any]) -> list[int]:
    """The topology's `numa_ports` projection for ONE role.

    The registry's `numa_ports` is the launcher's instance list MINUS the full
    instance (a role with no `full_instance_idx` has no full instance to
    remove). Kept identical to `stack_manifest._launcher_declarations`, which
    computes the same projection for the parity guard — if the two ever compute
    it differently, one of them is wrong about what the registry means.
    """
    instances = cfg.get("instances") or []
    full_idx = cfg.get("full_instance_idx")
    return [
        instance[1]
        for idx, instance in enumerate(instances)
        if full_idx is None or idx != full_idx
    ]


def _bind_role_to_topology(
    role: str, cfg: dict[str, Any], numa_config: dict[str, Any]
) -> str | None:
    """Resolve a registry `server_mode` row to the topology role it governs.

    Mirrors `stack_manifest.master_server_row` / `stack_priors._server_for_role`
    in reverse, with the same precedence: a row governs the topology role of the
    same name, else the role named by its `model_role`, else a role listed in
    its `shared_with` aliases. `None` = the topology has never heard of this
    row (which is only a defect if the row declares a fleet).
    """
    if role in numa_config:
        return role
    model_role = cfg.get("model_role")
    if isinstance(model_role, str) and model_role in numa_config:
        return model_role
    shared_with = cfg.get("shared_with")
    if isinstance(shared_with, list):
        for alias in shared_with:
            if isinstance(alias, str) and alias in numa_config:
                return alias
    return None


def _check_numa_ports_vs_topology(
    registry: dict[str, Any], numa_config: dict[str, Any] | None = None
) -> list[str]:
    """Cross-check `server_mode.<role>.numa_ports` against the NUMA topology.

    `numa_config` is injectable so a test can drive a synthetic topology; the
    default is the live `NUMA_CONFIG` the launcher itself imports.
    """
    errors: list[str] = []

    if numa_config is None:
        try:
            numa_config = _load_numa_config()
        except Exception as exc:  # noqa: BLE001 — report, never skip
            return [
                f"numa_ports cross-check could not load the declared topology "
                f"({type(exc).__name__}: {exc}). The check is NOT optional: "
                f"treat an unloadable topology as a failed registry, not as a "
                f"registry with nothing to disagree with."
            ]

    if not numa_config:
        return [
            "numa_ports cross-check: the declared NUMA topology is EMPTY. "
            "An empty topology makes every registry fleet vacuously consistent; "
            "refusing instead."
        ]

    server_mode = registry.get("server_mode") or {}
    server_roles = {k: v for k, v in server_mode.items() if isinstance(v, dict)}

    for role in sorted(server_roles):
        cfg = server_roles[role]
        declared_ports = cfg.get("numa_ports")
        declared_count = cfg.get("numa_instances")
        topo_role = _bind_role_to_topology(role, cfg, numa_config)

        if topo_role is None:
            if declared_ports is not None or declared_count is not None:
                errors.append(
                    f"role {role!r} declares numa_ports={declared_ports!r} / "
                    f"numa_instances={declared_count!r} but the NUMA topology "
                    f"(stack_topology.yaml numa_config) has NO entry for it, nor "
                    f"for its model_role/shared_with bindings. A fleet nothing "
                    f"launches is a phantom: either add the topology entry or "
                    f"delete the declaration."
                )
            continue

        topo_cfg = numa_config[topo_role]
        expected_ports = _topology_fleet_ports(topo_cfg)
        instance_ports = [instance[1] for instance in (topo_cfg.get("instances") or [])]
        where = (
            f"role {role!r}"
            if topo_role == role
            else f"role {role!r} (topology role {topo_role!r})"
        )

        if declared_ports is not None:
            if not isinstance(declared_ports, list) or not all(
                isinstance(p, int) and not isinstance(p, bool) for p in declared_ports
            ):
                errors.append(
                    f"{where}: numa_ports must be a list of ints, got "
                    f"{declared_ports!r}."
                )
            else:
                if len(set(declared_ports)) != len(declared_ports):
                    errors.append(
                        f"{where}: numa_ports {declared_ports!r} contains duplicate "
                        f"ports; one port is one llama-server instance."
                    )
                if sorted(declared_ports) != sorted(expected_ports):
                    errors.append(
                        f"{where}: numa_ports {sorted(declared_ports)!r} disagrees "
                        f"with the NUMA topology, which declares "
                        f"{sorted(expected_ports)!r} for {topo_role!r} "
                        f"(instances {instance_ports!r}, full_instance_idx="
                        f"{topo_cfg.get('full_instance_idx')!r}). "
                        f"stack_topology.yaml owns the fleet; the registry copy "
                        f"must be a projection of it, never an independent claim."
                    )
                if isinstance(declared_count, int) and not isinstance(declared_count, bool):
                    if declared_count != len(declared_ports):
                        errors.append(
                            f"{where}: numa_instances={declared_count} but "
                            f"numa_ports has {len(declared_ports)} entries "
                            f"({declared_ports!r}). numa_instances is the length "
                            f"of numa_ports, not a second opinion about it."
                        )
        elif declared_count is not None:
            if declared_count != len(expected_ports):
                errors.append(
                    f"{where}: numa_instances={declared_count!r} but the NUMA "
                    f"topology declares {len(expected_ports)} non-full "
                    f"instance(s) {expected_ports!r} for {topo_role!r}."
                )
        elif len(instance_ports) > 1:
            errors.append(
                f"{where}: the NUMA topology declares {len(instance_ports)} "
                f"instances {instance_ports!r} but the registry declares no "
                f"numa_ports. Consumers that fan out over numa_ports "
                f"(model_descriptors, stack_priors, eval_tower) then see ONE "
                f"backend and collapse to it silently — declare "
                f"numa_ports: {expected_ports!r}."
            )

        primary = cfg.get("port")
        if isinstance(primary, int) and not isinstance(primary, bool) and instance_ports:
            full_idx = topo_cfg.get("full_instance_idx")
            if full_idx is not None and 0 <= full_idx < len(instance_ports):
                if primary != instance_ports[full_idx]:
                    errors.append(
                        f"{where}: port {primary} is not the topology's full "
                        f"instance port {instance_ports[full_idx]} "
                        f"(full_instance_idx={full_idx}, instances "
                        f"{instance_ports!r}). The primary port must address the "
                        f"full-speed instance."
                    )
            elif primary not in instance_ports:
                errors.append(
                    f"{where}: port {primary} is not one of the topology's "
                    f"instance ports {instance_ports!r} for {topo_role!r}."
                )

    return errors


def _check_drafter_selection(
    registry: dict[str, Any], topology_path: Path | None = None
) -> list[str]:
    """DRAFT-SEL-1 over the COMPILED lean registry (the file cmd_start validates).

    * topology selects a drafter the model's `drafters` list does not contain -> error
    * a launching server whose model lists >1 drafter and has no selection -> error
    * the lean's projected drafter differs from what topology selects, or its spec
      fields differ from the master recipe (stale or hand-edited lean) -> error

    Hand-carried drafter fields are caught one hop earlier, in compile_lean, because
    after projection they are indistinguishable from the projection's own output.
    """
    from src.registry.drafter_selection import (
        DrafterSelectionError,
        check_lean,
        load_drafter_selection,
    )

    try:
        selection = load_drafter_selection(topology_path)
    except DrafterSelectionError as exc:
        return [f"drafter_selection: {exc}"]
    return check_lean(registry, selection)


def validate_all(registry_path: str | Path) -> list[str]:
    """Run all checks. Returns list of error strings. Empty list = OK.

    Caller should print the errors and exit non-zero if any returned.
    """
    path = Path(registry_path)
    if not path.exists():
        return [f"registry not found: {path}"]

    registry = _strict_yaml_load(path)  # may raise RegistryValidationError on dup keys

    errors: list[str] = []
    errors += _check_cross_section_conflicts(registry)
    errors += _check_gguf_port_consistency(registry)
    errors += _check_numa_ports_vs_topology(registry)
    errors += _check_drafter_selection(registry)
    return errors


def validate_or_raise(registry_path: str | Path) -> None:
    """Validate; on any error, raise RegistryValidationError with combined message."""
    errors = validate_all(registry_path)
    if errors:
        joined = "\n  - ".join(errors)
        raise RegistryValidationError(
            f"Registry validation failed for {registry_path}:\n  - {joined}"
        )


if __name__ == "__main__":
    # `sys` is imported at module scope (the topology cross-check needs it on
    # sys.path); the former local `import sys` here would shadow it.
    target = sys.argv[1] if len(sys.argv) > 1 else (
        str(_REPO_ROOT / "orchestration/model_registry.yaml")
    )
    try:
        errors = validate_all(target)
    except RegistryValidationError as exc:
        print(f"FAIL: {exc}")
        sys.exit(2)
    if errors:
        print(f"FAIL: {len(errors)} validation error(s) in {target}:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"OK: {target}")
