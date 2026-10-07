"""Compile a lean orchestrator registry from the master research registry.

Background (2026-05-09): Today's recovery hit a bug because the orchestrator's
checked-in `model_registry.yaml` had drifted from the master at
`epyc-inference-research/orchestration/model_registry.yaml`. The validator
catches in-file inconsistencies but the master/orchestrator coupling itself
was manually synchronized — silent drift was inevitable.

This module compiles a lean view from `(master + active_role_set)` at every
stack launch. The lean view contains exactly the roles the orchestrator will
launch plus their transitive dependencies (drafts, aliases). Small registry
metadata sections used by scripts/procedures are preserved, while bulky
non-active role/server catalogue records are dropped.

Cache-invalidates by SHA-256 of `(master content + sorted active role names)`.
If neither side has changed, the cached compile output is reused.

Wired into `orchestrator_stack.py:cmd_start` before the validator gate. The
auto-generated lean output replaces the orchestrator's `model_registry.yaml`.
A header banner makes the auto-generated nature visible to anyone opening the
file.

Escape hatch: setting `ORCHESTRATOR_REGISTRY_NO_COMPILE=1` skips the compile
step and falls back to the on-disk file (useful during a master-registry
schema change when you need to inspect or re-sync generated output).
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.registry.drafter_selection import (
    DEFAULT_TOPOLOGY_PATH,
    DrafterSelectionError,
    load_drafter_selection,
    project as project_drafters,
    resolve as resolve_drafters,
    selection_cache_bytes,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASTER_PATH = Path(
    "/mnt/raid0/llm/epyc-inference-research/orchestration/model_registry.yaml"
)

log = logging.getLogger("registry.compiler")


# Top-level sections in the master that the orchestrator runtime DOES need.
# Anything not listed is dropped from the lean output.
_KEEP_SECTIONS_FULL = (
    "runtime_defaults",
    "worker_pool",
    "process_layout",
    "escalation_chains",
    "routing_hints",
    "command_templates",
    "dflash_drafters",
    "deprecated_models",
    "observations",
    "optimized_params",
    "runtime_quirks",
    "kernel_audits",
    # 2026-08-02: was ABSENT, so the master's own canonical spec-recipe block was
    # dropped from every compiled lean registry. `stack_priors.py:1178,1634` refer
    # to it as *"the canonical `speculative_decoding_policy` block in the master
    # registry"* while the compiled artifact did not contain it and nothing read it
    # — an authoritative-sounding declaration that was inert. Two sources of truth
    # for the spec recipe, one of them decorative, is how `ngram-mod` came to sit in
    # per-role `acceleration` blocks while the policy said otherwise.
    "speculative_decoding_policy",
)

# Top-level master sections that are DELIBERATELY not carried into the lean output,
# with the reason. Anything in master that is in neither this set nor
# _KEEP_SECTIONS_FULL / _FILTER_ROLE_KEYED is reported by _warn_unlisted_sections:
# a hand-written keep-list silently drops whatever nobody remembered to add, which
# is exactly how the block above went missing. The drop must be a decision on the
# record, not an omission.
_INTENTIONALLY_DROPPED_SECTIONS: dict[str, str] = {
    "schema_version": "metadata; the lean artifact carries its own",
    "last_updated": "metadata",
    "notes": "human prose, not runtime input",
}


def _warn_unlisted_sections(master: dict[str, Any]) -> list[str]:
    """Return a warning per master section the compiler neither keeps nor declines.

    Silence here is the defect: the compiler cannot know that a section it has
    never heard of is unimportant, and the only evidence of a drop is its absence
    from an artifact nobody diffs.
    """
    known = (
        set(_KEEP_SECTIONS_FULL)
        | set(_FILTER_ROLE_KEYED)
        | set(_INTENTIONALLY_DROPPED_SECTIONS)
    )
    return [
        f"master registry section {name!r} is dropped from the lean output: it is in "
        f"neither _KEEP_SECTIONS_FULL nor _INTENTIONALLY_DROPPED_SECTIONS. Add it to "
        f"one of them so the decision is on the record."
        for name in sorted(master)
        if name not in known
    ]
# Sections that are role-keyed dicts and need filtering.
_FILTER_ROLE_KEYED = ("server_mode", "roles")
_RETIRED_TIMEOUT_ROLE_ALIASES = frozenset({"worker_code"})


def _server_mode_entry_roles(entry_name: str, entry: Any) -> set[str]:
    """Return all role names covered by a server_mode entry."""
    roles = {entry_name}
    if not isinstance(entry, dict):
        return roles

    model_role = entry.get("model_role")
    if isinstance(model_role, str):
        roles.add(model_role)

    shared_with = entry.get("shared_with")
    if isinstance(shared_with, list):
        roles.update(str(role) for role in shared_with)

    return roles


def _filter_server_mode(server_mode: dict[str, Any], needed: set[str]) -> dict[str, Any]:
    """Keep server records that directly or indirectly serve needed roles."""
    return {
        name: entry
        for name, entry in server_mode.items()
        if _server_mode_entry_roles(name, entry) & needed
    }


def _normalize_runtime_defaults(runtime_defaults: Any) -> Any:
    """Return runtime defaults after dropping retired compatibility aliases."""
    if not isinstance(runtime_defaults, dict):
        return runtime_defaults

    normalized = dict(runtime_defaults)
    timeouts = normalized.get("timeouts")
    if isinstance(timeouts, dict):
        normalized_timeouts = dict(timeouts)
        role_timeouts = normalized_timeouts.get("roles")
        if isinstance(role_timeouts, dict):
            normalized_timeouts["roles"] = {
                role: timeout
                for role, timeout in role_timeouts.items()
                if role not in _RETIRED_TIMEOUT_ROLE_ALIASES
            }
        normalized["timeouts"] = normalized_timeouts
    return normalized


def cache_key(
    master_path: Path,
    active_roles: set[str],
    topology_path: Path | None = None,
) -> str:
    """SHA-256 of master bytes + sorted active role names + drafter selection.

    Recompiles when ANY input changes. The drafter selection (stack_topology.yaml
    `drafter_selection`) is a compile INPUT since DRAFT-SEL-1: without it in the key,
    a topology-only drafter change would hit the cache and never reach the lean view.
    Only that section is hashed, so NUMA-only topology edits do not churn the lean.
    """
    h = hashlib.sha256()
    h.update(master_path.read_bytes())
    h.update(b"\x00")  # separator
    for r in sorted(active_roles):
        h.update(r.encode("utf-8"))
        h.update(b"\x00")
    h.update(b"drafter_selection\x00")
    h.update(selection_cache_bytes(load_drafter_selection(topology_path)))
    return h.hexdigest()


def _resolve_role_dependencies(master: dict, active_roles: set[str]) -> set[str]:
    """Walk role definitions to find transitive deps (draft models, aliases).

    A role's `acceleration.draft_role` references another role by name (e.g.
    `architect_general.acceleration.draft_role = "draft_qwen35_0_8b_q8_0"`).
    The draft role MUST be in the lean output too or `RegistryLoader.get_role`
    will fail when the launcher tries to look it up.
    """
    resolved = set(active_roles)
    roles_section = master.get("roles") or {}
    server_mode = master.get("server_mode") or {}

    # Iterate until fixed point — a draft role might itself reference another role.
    changed = True
    while changed:
        changed = False
        for r in list(resolved):
            for source in (roles_section.get(r), server_mode.get(r)):
                if not isinstance(source, dict):
                    continue
                accel = source.get("acceleration")
                if isinstance(accel, dict):
                    dr = accel.get("draft_role")
                    if dr and dr not in resolved:
                        resolved.add(dr)
                        changed = True
                # `model_role` field cross-refs a role definition (architect_general
                # in server_mode points at qwen35_122b_q4km — both must survive).
                mr = source.get("model_role")
                if mr and mr not in resolved:
                    resolved.add(mr)
                    changed = True
                # `shared_with` lists — defensively include
                sw = source.get("shared_with")
                if isinstance(sw, list):
                    for s in sw:
                        if s not in resolved:
                            resolved.add(s)
                            changed = True
    return resolved


def launcher_tenant_roles(launch_meta: dict[str, Any]) -> set[str]:
    """Return registry TENANT roles named by launcher-only entries.

    gpu-serving-tie-in P2-6 (P0-1 compile contract): a ``launcher_only`` entry
    may name a registry role via the optional ``tenant_role`` meta key. The
    launcher entry itself stays out of the compile inputs (it is a process
    identity, not a model-routing role), but the named TENANT role must flow
    through lean-registry / descriptor / stack-prior compilation so an
    explicitly-requested launcher-only start can resolve its launch record.
    No production ``ROLE_LAUNCH_META`` entry carries ``tenant_role`` today —
    with the key absent everywhere this returns an empty set and every compile
    output is byte-identical to the pre-key behavior (inert-by-construction).
    """
    tenants: set[str] = set()
    for meta in launch_meta.values():
        if not isinstance(meta, dict) or meta.get("launcher_only") is not True:
            continue
        tenant = meta.get("tenant_role")
        if isinstance(tenant, str) and tenant:
            tenants.add(tenant)
    return tenants


def active_roles_from_launch_meta(launch_meta: dict[str, Any]) -> set[str]:
    """Return compile input roles from launcher metadata, including aliases.

    `ROLE_LAUNCH_META` keys are the primary launch targets. Some production
    roles are aliases attached to the first N instances of a primary target
    (`shared_with_first_n`), so they do not appear as top-level launch keys.
    The lean registry still needs their `roles.X` / `server_mode.X` records for
    routing, timeout, descriptor, and attestation consumers.

    ``launcher_only`` entries are skipped (process identities, not roles), but
    a launcher-only entry may name a registry TENANT role via the optional
    ``tenant_role`` key (``launcher_tenant_roles``); the tenant compiles
    through like an active role WITHOUT joining the live stack start-set.
    """
    active: set[str] = set()
    for role, meta in launch_meta.items():
        if not isinstance(meta, dict):
            active.add(str(role))
            continue
        if meta.get("launcher_only") is True:
            continue
        active.add(str(role))
        aliases = meta.get("shared_with_first_n")
        if isinstance(aliases, list):
            active.update(str(alias) for alias in aliases)
    active.update(launcher_tenant_roles(launch_meta))
    return active


def compile_lean(
    master_path: Path,
    active_roles: set[str],
    topology_path: Path | None = None,
) -> dict:
    """Project master registry through the active stack manifest AND the topology's
    drafter selection.

    Returns a dict in the same shape RegistryLoader expects. Raises
    DrafterSelectionError when the master/topology pair does not determine exactly
    one drafter for every launching server whose model declares `drafters`
    (src/registry/drafter_selection.py has the rules).
    """
    with master_path.open("r", encoding="utf-8") as f:
        master = yaml.safe_load(f)
    if not isinstance(master, dict):
        raise ValueError(f"master registry at {master_path} did not parse to a dict")

    needed = _resolve_role_dependencies(master, active_roles)

    for warning in _warn_unlisted_sections(master):
        log.warning("%s", warning)

    out: dict[str, Any] = {}

    # Full-keep sections — small, no filtering needed.
    for section in _KEEP_SECTIONS_FULL:
        if section in master:
            value = master[section]
            if section == "runtime_defaults":
                value = _normalize_runtime_defaults(value)
            out[section] = value

    # Role-keyed sections — filter to needed roles only. `server_mode` entries
    # may be keyed by backing process name rather than route role name
    # (`worker` -> `worker_general`), so inspect entry metadata too.
    for section in _FILTER_ROLE_KEYED:
        src = master.get(section)
        if not isinstance(src, dict):
            continue
        if section == "server_mode":
            out[section] = _filter_server_mode(src, needed)
        else:
            out[section] = {k: v for k, v in src.items() if k in needed}

    # DRAFT-SEL-1: the drafter is SELECTED by topology from the master's list, never
    # hand-carried per role. Resolution runs over the launching servers only.
    selection = load_drafter_selection(topology_path)
    resolution = resolve_drafters(master, active_roles, selection)
    for warning in resolution.warnings:
        log.warning("%s", warning)
    if resolution.errors:
        raise DrafterSelectionError(
            "drafter selection does not resolve:\n  - " + "\n  - ".join(resolution.errors)
        )
    project_drafters(out, master, resolution)

    return out


def _format_header_banner(
    master_path: Path,
    active_roles: set[str],
    cache_key_value: str,
    topology_path: Path | None = None,
) -> str:
    """Top-of-file comment block for the compiled output."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    role_list = ", ".join(sorted(active_roles))
    resolved_master = master_path.expanduser().resolve(strict=False)
    canonical_master = DEFAULT_MASTER_PATH.resolve(strict=False)
    if resolved_master == canonical_master:
        master_source = f"# canonical master registry at {resolved_master}\n"
    else:
        master_source = (
            f"# noncanonical master input at {resolved_master}\n"
            f"# canonical source of truth: {canonical_master}\n"
        )
    return (
        "# =============================================================================\n"
        "# AUTO-GENERATED — MASTER-COMPILED RUNTIME VIEW.\n"
        "#\n"
        "# This file is compiled at every `orchestrator_stack.py start` from the master\n"
        + master_source
        + f"# with drafter selection from {topology_path or DEFAULT_TOPOLOGY_PATH}\n"
        "# by src/registry/registry_compiler.py.\n"
        "#\n"
        "# Runtime stack truth lives in the MASTER registry. The next start detects master\n"
        "# changes by cache-key mismatch and regenerates this lean runtime view. To\n"
        "# edit registry data, change the master source; this generated view is disposable.\n"
        "# temporarily skip the compile (e.g. during a master schema change), set\n"
        "# ORCHESTRATOR_REGISTRY_NO_COMPILE=1.\n"
        "#\n"
        f"# Compiled at: {now}\n"
        f"# Cache key:   {cache_key_value[:16]}...\n"
        f"# Active roles: {role_list}\n"
        "# =============================================================================\n"
        "\n"
    )


def load_or_compile(
    master_path: Path,
    active_roles: set[str],
    output_path: Path,
    cache_key_path: Path,
    topology_path: Path | None = None,
) -> dict:
    """Cache-aware compile.

    On cache hit (key matches) returns the existing `output_path` parsed.
    On miss, recompiles, writes `output_path`, updates `cache_key_path`.
    """
    if os.environ.get("ORCHESTRATOR_REGISTRY_NO_COMPILE") == "1":
        # Escape hatch — read whatever's on disk, no compile.
        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        raise FileNotFoundError(
            f"ORCHESTRATOR_REGISTRY_NO_COMPILE=1 set but {output_path} does not exist"
        )

    if not master_path.exists():
        raise FileNotFoundError(f"master registry not found: {master_path}")

    current_key = cache_key(master_path, active_roles, topology_path)
    cached_key = (
        cache_key_path.read_text().strip()
        if cache_key_path.exists()
        else None
    )

    if cached_key == current_key and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            cached = yaml.safe_load(f)
        if isinstance(cached, dict):
            return cached
        # Cached file unparseable — fall through to recompile.

    # Compile, then avoid rewriting an already-current generated file when only
    # the external cache-key file is missing or stale. This keeps default-on
    # startup from dirtying the tracked registry via timestamp-only banner churn.
    compiled = compile_lean(master_path, active_roles, topology_path)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing = yaml.safe_load(f)
        if existing == compiled:
            cache_key_path.parent.mkdir(parents=True, exist_ok=True)
            cache_key_path.write_text(current_key)
            return compiled

    banner = _format_header_banner(master_path, active_roles, current_key, topology_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(banner)
        yaml.safe_dump(
            compiled,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=200,
        )
    cache_key_path.parent.mkdir(parents=True, exist_ok=True)
    cache_key_path.write_text(current_key)
    return compiled


# --- CLI for diagnostic / dry-run inspection -------------------------------

def _main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Compile lean orchestrator registry")
    p.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER_PATH,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "orchestration/model_registry.yaml",
    )
    p.add_argument(
        "--topology",
        type=Path,
        default=DEFAULT_TOPOLOGY_PATH,
        help="stack_topology.yaml whose `drafter_selection` picks each server's drafter",
    )
    p.add_argument(
        "--cache-key",
        type=Path,
        default=_REPO_ROOT / "orchestration/.lean_cache_key",
    )
    p.add_argument(
        "--roles",
        nargs="+",
        help="Explicit active role list. If omitted, imports ROLE_LAUNCH_META keys.",
    )
    p.add_argument("--dry-run", action="store_true", help="Compile to stdout, do not write")
    p.add_argument("--force", action="store_true", help="Recompile even on cache hit")
    args = p.parse_args()

    if args.roles:
        active = set(args.roles)
    else:
        # Import the declarative manifest, not orchestrator_stack.py: the stack
        # CLI owns argparse side effects at module import time.
        sys.path.insert(0, str(_REPO_ROOT))
        from scripts.server.stack_manifest import ROLE_LAUNCH_META  # type: ignore[import]
        active = active_roles_from_launch_meta(ROLE_LAUNCH_META)

    if args.dry_run:
        compiled = compile_lean(args.master, active, args.topology)
        yaml.safe_dump(compiled, sys.stdout, sort_keys=False, default_flow_style=False)
        return 0

    if args.force and args.cache_key.exists():
        args.cache_key.unlink()

    out = load_or_compile(args.master, active, args.output, args.cache_key, args.topology)
    print(f"OK: {len(out.get('roles', {}))} roles in compiled output")
    print(f"  master:    {args.master}")
    print(f"  output:    {args.output}")
    print(f"  topology:  {args.topology}")
    print(f"  cache key: {args.cache_key}")
    print(f"  active:    {sorted(active)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
