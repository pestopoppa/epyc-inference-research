"""Compile derived stack priors from registry and model descriptors.

This module is intentionally additive: existing consumers can migrate to the
generated artifact one by one instead of re-parsing scattered registry comments
and hardcoded role tables.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

_LOGGER = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "orchestration" / "model_registry.yaml"
DEFAULT_DESCRIPTORS = REPO_ROOT / "orchestration" / "model_descriptors.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
DEFAULT_STACK_MANIFEST = REPO_ROOT / "scripts" / "server" / "stack_manifest.py"
DEFAULT_STACK_NUMA = REPO_ROOT / "scripts" / "server" / "stack_numa.py"
# 2026-08-01: the launcher's configuration moved OUT of the two .py files above
# and into these declared artifacts; those modules are now thin loaders. Pinning
# only the loaders would leave the provenance chain pointing at files that no
# longer contain the facts — edit launch_manifest.yaml and every pinned hash
# would still match, so a stale derived layer would look fresh.
# The byte-hash pin caught exactly that failure earlier today when
# orchestrator_stack.py changed; it must keep covering the data now that the data
# has moved.
DEFAULT_LAUNCH_MANIFEST = REPO_ROOT / "orchestration" / "launch_manifest.yaml"
DEFAULT_STACK_TOPOLOGY = REPO_ROOT / "orchestration" / "stack_topology.yaml"
DEFAULT_ORCHESTRATOR_STACK = REPO_ROOT / "scripts" / "server" / "orchestrator_stack.py"
DEFAULT_STACK_PATHS = REPO_ROOT / "scripts" / "server" / "stack_paths.py"
DEFAULT_STACK_RUNTIME = REPO_ROOT / "scripts" / "server" / "stack_runtime.py"
PRECEDENCE_SPEC = REPO_ROOT / "docs" / "reference" / "stack-truth-precedence.md"
DEFAULT_MODELS_DIR = Path("/mnt/raid0/llm/models")
DEFAULT_MODEL_BASE_DIR = Path("/mnt/raid0/llm/models")

STACK_PRIORS_VERSION = 4
REQUIRED_TOP_LEVEL_FIELDS = (
    "stack_priors_version",
    "contract",
    "compiled_at",
    "status",
    "coverage_scope",
    "precedence_spec",
    "source_artifacts",
    "roles",
    "known_global_gaps",
)
REQUIRED_ROLE_FIELDS = (
    "role",
    "deployment_status",
    "status",
    "model_id",
    "display_name",
    "serving",
    "priors",
    "acceleration",
    "model",
    "evidence",
    "known_gaps",
)
REQUIRED_SERVING_FIELDS = (
    "endpoint",
    "server_role",
    "binding",
    "ports",
    "slots",
    "tier",
    "effective_context_tokens",
    "binary",
    "binary_dir",
    "numa_policy",
    "shared_mmap",
    "launch",
)
REQUIRED_LAUNCH_FIELDS = (
    "entries",
    "primary_roles",
    "modes",
    "requirements",
    "runtime",
)
REQUIRED_PRIOR_FIELDS = (
    "throughput_tps",
    "quality_overall",
    "memory_cost",
)

RESIDENCY_COST = {"hot": 1.0, "warm": 2.0, "cold": 3.0}

# Canonical model-memory thresholds (GB) for the projected per-role policy hints.
# These mirror the runtime consumers that currently re-derive the same
# classifications locally (src/runtime/inference_tap.py safe-non-stream default,
# src/graph/approval_gate.py high-cost gate); projecting them here lets those
# tables become fallback/override only.
POLICY_TAP_SAFE_NON_STREAM_MIN_MEM_GB = 64.0
POLICY_HIGH_COST_MIN_MEM_GB = 60.0


class StackPriorsCompileError(ValueError):
    """Stack-prior compilation found unresolved live-role gaps."""

    def __init__(self, gaps_by_role: dict[str, list[str]]) -> None:
        self.gaps_by_role = gaps_by_role
        lines = ["Stack prior compilation refused unresolved role gaps:"]
        for role, gaps in sorted(gaps_by_role.items()):
            for gap in gaps:
                lines.append(f"  - {role}: {gap}")
        super().__init__("\n".join(lines))


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return loaded


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_commit(path: Path) -> str | None:
    resolved = path.resolve()
    parents = (resolved,) if resolved.is_dir() else (resolved.parent,)
    for parent in parents[0].parents:
        if (parent / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                return None
    return None


def _portable_path(path: Path) -> str:
    """Repo-relative for files in this checkout, absolute otherwise.

    Every checkout-derived path this compiler writes goes through here, so the
    committed artifact is byte-identical whichever worktree compiled it (see
    model_descriptors.portable_source_path for the defect this removes).
    """
    from src.registry.model_descriptors import portable_source_path

    return portable_source_path(path)


def _source_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": _portable_path(path),
        "sha256": _sha256(path),
        "repo_commit": _repo_commit(path),
    }


def stack_priors_contract() -> dict[str, Any]:
    """Return the versioned consumer contract embedded in generated priors."""
    return {
        "schema": "epyc.stack_priors",
        "version": STACK_PRIORS_VERSION,
        "required_top_level_fields": list(REQUIRED_TOP_LEVEL_FIELDS),
        "required_role_fields": list(REQUIRED_ROLE_FIELDS),
        "required_serving_fields": list(REQUIRED_SERVING_FIELDS),
        "required_launch_fields": list(REQUIRED_LAUNCH_FIELDS),
        "required_prior_fields": list(REQUIRED_PRIOR_FIELDS),
        "fallback_policy": (
            "Consumers may use local fallback values only as explicit degraded "
            "mode when this generated contract is missing or invalid."
        ),
    }


def validate_stack_priors_contract(priors: dict[str, Any]) -> list[str]:
    """Validate the generated stack-priors consumer contract shape.

    This intentionally checks structure, not semantic freshness. Source hashes,
    retired live roles, and strict known-gap policy remain in stack_change_guard.
    """
    errors: list[str] = []
    if priors.get("stack_priors_version") != STACK_PRIORS_VERSION:
        errors.append(
            f"stack_priors_version must be {STACK_PRIORS_VERSION}, "
            f"got {priors.get('stack_priors_version')!r}"
        )

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in priors:
            errors.append(f"missing top-level stack-prior field: {field}")

    contract = priors.get("contract")
    if not isinstance(contract, dict):
        errors.append("stack priors artifact has no mapping-valued contract section")
    elif contract.get("version") != STACK_PRIORS_VERSION:
        errors.append(
            f"stack-prior contract version must be {STACK_PRIORS_VERSION}, "
            f"got {contract.get('version')!r}"
        )

    roles = priors.get("roles")
    if not isinstance(roles, dict):
        errors.append("stack priors artifact has no mapping-valued roles section")
        return errors

    for role, record in sorted(roles.items()):
        if not isinstance(record, dict):
            errors.append(f"role {role!r} record is not a mapping")
            continue
        for field in REQUIRED_ROLE_FIELDS:
            if field not in record:
                errors.append(f"role {role!r} is missing contract field {field!r}")
        serving = record.get("serving")
        if not isinstance(serving, dict):
            errors.append(f"role {role!r} serving is not a mapping")
        else:
            for field in REQUIRED_SERVING_FIELDS:
                if field not in serving:
                    errors.append(f"role {role!r} serving is missing field {field!r}")
            launch = serving.get("launch")
            if not isinstance(launch, dict):
                errors.append(f"role {role!r} serving.launch is not a mapping")
            else:
                for field in REQUIRED_LAUNCH_FIELDS:
                    if field not in launch:
                        errors.append(f"role {role!r} serving.launch is missing field {field!r}")
        priors_block = record.get("priors")
        if not isinstance(priors_block, dict):
            errors.append(f"role {role!r} priors is not a mapping")
        else:
            for field in REQUIRED_PRIOR_FIELDS:
                if field not in priors_block:
                    errors.append(f"role {role!r} priors is missing field {field!r}")
        known_gaps = record.get("known_gaps")
        if not isinstance(known_gaps, list):
            errors.append(f"role {role!r} known_gaps must be a list")
    return errors


def load_stack_priors_artifact(path: Path = DEFAULT_OUTPUT) -> dict[str, Any] | None:
    """Load a generated stack-priors artifact for runtime consumers.

    Compilation and validation paths should use the stricter helpers in this
    module. Runtime consumers use this fail-closed loader so a missing or
    malformed generated artifact falls back to their explicit degraded modes.
    """
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    return loaded if isinstance(loaded, dict) else None


def live_stack_role_records(path: Path = DEFAULT_OUTPUT) -> dict[str, dict[str, Any]]:
    """Return live stack-prior role records keyed by role name."""
    artifact = load_stack_priors_artifact(path)
    if artifact is None:
        return {}
    roles = artifact.get("roles")
    if not isinstance(roles, dict):
        return {}

    live: dict[str, dict[str, Any]] = {}
    for role, record in roles.items():
        if not isinstance(role, str):
            continue
        if not isinstance(record, dict) or record.get("deployment_status") != "live_stack":
            continue
        live[role] = record
    return live


def launcher_tenant_role_records(path: Path = DEFAULT_OUTPUT) -> dict[str, dict[str, Any]]:
    """Return launcher-tenant stack-prior role records keyed by role name.

    gpu-serving-tie-in P2-6 (P0-1): TENANT roles named by ``launcher_only``
    launch-meta entries (``tenant_role`` key) compile with
    ``deployment_status: launcher_tenant``. They are deliberately EXCLUDED from
    ``live_stack_role_records`` so routing/lock/attestation consumers never see
    them; only an explicitly-requested launcher-only start resolves them (via
    ``orchestrator_stack._stack_prior_launch``). Empty today: no production
    launch-meta entry carries ``tenant_role``.
    """
    artifact = load_stack_priors_artifact(path)
    if artifact is None:
        return {}
    roles = artifact.get("roles")
    if not isinstance(roles, dict):
        return {}

    tenants: dict[str, dict[str, Any]] = {}
    for role, record in roles.items():
        if not isinstance(role, str):
            continue
        if not isinstance(record, dict) or record.get("deployment_status") != "launcher_tenant":
            continue
        tenants[role] = record
    return tenants


def canonical_stack_role_id(role_name: str) -> str | None:
    """Return the canonical role ID for a known stack role or alias."""
    from src.roles import Role

    role = Role.from_string(str(role_name))
    return role.value if role is not None else None


def live_stack_role_ids(
    path: Path = DEFAULT_OUTPUT,
    *,
    preferred_order: Sequence[str] = (),
) -> list[str]:
    """Return canonical live stack role IDs in a stable preferred order."""
    live_records = live_stack_role_records(path)
    if not live_records:
        return []

    live: list[str] = []
    for role_id, record in live_records.items():
        raw_role = record.get("role") if isinstance(record, dict) else None
        role_name = raw_role if isinstance(raw_role, str) and raw_role else role_id
        live.append(canonical_stack_role_id(role_name) or str(role_name))

    live = list(dict.fromkeys(live))
    if not live:
        return []

    live_set = set(live)
    ordered: list[str] = []
    seen: set[str] = set()
    for role in preferred_order:
        canonical = canonical_stack_role_id(str(role)) or str(role)
        if canonical in live_set and canonical not in seen:
            ordered.append(canonical)
            seen.add(canonical)
    ordered.extend(role for role in live if role not in seen)
    return ordered


def stack_prior_serving(record: dict[str, Any]) -> dict[str, Any]:
    """Return the mapping-valued ``serving`` block from a role record."""
    serving = record.get("serving")
    return serving if isinstance(serving, dict) else {}


def stack_prior_launch(record: dict[str, Any]) -> dict[str, Any]:
    """Return the mapping-valued ``serving.launch`` block from a role record."""
    launch = stack_prior_serving(record).get("launch")
    return launch if isinstance(launch, dict) else {}


def stack_prior_launch_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return mapping-valued launch entries from a stack-prior role record."""
    entries = stack_prior_launch(record).get("entries")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def stack_prior_launch_modes(record: dict[str, Any]) -> set[str]:
    """Return string launch modes from a stack-prior role record."""
    modes = stack_prior_launch(record).get("modes")
    if not isinstance(modes, list):
        return set()
    return {mode for mode in modes if isinstance(mode, str)}


def stack_prior_uses_shared_worker_launch(record: dict[str, Any]) -> bool:
    """Return True when stack-prior launch metadata identifies shared worker use."""
    if "worker_pool" in stack_prior_launch_modes(record):
        return True
    for entry in stack_prior_launch_entries(record):
        if entry.get("mode") == "worker_pool":
            return True
        if entry.get("vision_type") == "worker":
            return True
    return False


def stack_prior_model_mem_gb(record: dict[str, Any]) -> float | None:
    """Return numeric model memory from a stack-prior role record, if present."""
    model = record.get("model")
    mem_gb = model.get("mem_gb") if isinstance(model, dict) else None
    if not isinstance(mem_gb, (int, float)):
        return None
    return float(mem_gb)


def live_stack_lock_role_sets(
    path: Path = DEFAULT_OUTPUT,
) -> tuple[frozenset[str], frozenset[str]] | None:
    """Derive exclusive/shared lock role sets from generated live stack priors."""
    roles = live_stack_role_records(path)
    if not roles:
        return None

    heavy: set[str] = set()
    light: set[str] = set()
    for role, record in roles.items():
        if stack_prior_uses_shared_worker_launch(record):
            light.add(role)
        else:
            heavy.add(role)

    if not heavy and not light:
        return None
    return frozenset(heavy), frozenset(light)


def live_stack_safe_non_stream_roles(
    path: Path = DEFAULT_OUTPUT,
    *,
    min_mem_gb: float,
) -> frozenset[str] | None:
    """Derive safe-mode non-stream roles from generated live stack-prior memory."""
    roles = live_stack_role_records(path)
    if not roles:
        return None

    threshold = max(0.0, float(min_mem_gb))
    derived: set[str] = set()
    saw_live_memory = False
    for role, record in roles.items():
        mem_gb = stack_prior_model_mem_gb(record)
        if mem_gb is None:
            continue
        saw_live_memory = True
        if mem_gb >= threshold:
            derived.add(role)

    if not saw_live_memory:
        return None
    return frozenset(derived)


def stack_prior_endpoint_port(serving: dict[str, Any]) -> int | None:
    """Return the endpoint port from a stack-prior serving block, if present."""
    endpoint = serving.get("endpoint")
    if not isinstance(endpoint, str):
        return None
    return urlparse(endpoint).port


def stack_prior_serving_ports(serving: dict[str, Any]) -> list[int]:
    """Return integer serving ports from a stack-prior serving block."""
    ports = serving.get("ports")
    if not isinstance(ports, list):
        return []
    return [port for port in ports if isinstance(port, int)]


def stack_prior_primary_port(serving: dict[str, Any]) -> int | None:
    """Return the endpoint port, falling back to the first declared serving port."""
    try:
        endpoint_port = stack_prior_endpoint_port(serving)
    except ValueError:
        endpoint_port = None
    if endpoint_port is not None:
        return endpoint_port
    for port in stack_prior_serving_ports(serving):
        return port
    return None


def stack_prior_serving_url_value(serving: dict[str, Any]) -> str | None:
    """Return the config-compatible URL value for a serving block."""
    ports = stack_prior_serving_ports(serving)
    if ports:
        urls = [f"http://localhost:{port}" for port in ports]
        if len(urls) > 1:
            urls[0] = f"full:{urls[0]}"
        return ",".join(urls)
    endpoint = serving.get("endpoint")
    return endpoint if isinstance(endpoint, str) and endpoint.startswith("http") else None


def live_stack_serving_url_values(path: Path = DEFAULT_OUTPUT) -> dict[str, str]:
    """Return config-compatible URL values keyed by live stack role."""
    urls: dict[str, str] = {}
    for role, record in live_stack_role_records(path).items():
        url = stack_prior_serving_url_value(stack_prior_serving(record))
        if url:
            urls[role] = url
    return urls


def live_stack_serving_slot_limits(path: Path = DEFAULT_OUTPUT) -> dict[str, int]:
    """Return per-serving-URL admission slot limits from live stack priors.

    2026-08-02 — PER PORT, NOT PER ROLE. This used to read the role-level
    `serving.slots` and fold it over every one of the role's ports with `max()`.
    Two consequences, both wrong once instances stopped being identical:

      * all three frontdoor endpoints got ONE limit, so a per-instance admission
        limit was not expressible at all — the 96-core full and the two 48-core
        halves were told to admit the same number of requests;
      * the number it folded was `serving.slots`, while the launcher emitted
        `runtime.cache.slots` for `-np`. Those two disagreed (2 vs 1 for
        frontdoor and architect_critic), so this module's stated purpose —
        "Aligned admission limits with llama-server slot counts (no idle slots)",
        src/api/admission.py — was false on the roles it mattered most for.

    Both are fixed by reading the per-instance `slots` now stamped on each launch
    ENTRY: it is the exact number the launcher passes to `-np` for that port, so
    `admission_limit == -np` holds instance by instance. The `max()` fold is kept
    ONLY for genuine collisions — two roles resolving to the same URL, e.g. an
    alias and its host — where the larger of two equal numbers is still that
    number, and a real disagreement should not silently under-admit the process.

    `serving.slots` remains the fallback for a record with no per-entry value
    (launcher-only tenants, or a role compiled before this field existed).
    """
    limits: dict[str, int] = {}

    def _bump(url: str, value: int) -> None:
        limits[url] = max(value, limits.get(url, 0))

    for record in live_stack_role_records(path).values():
        serving = stack_prior_serving(record)
        role_slots = serving.get("slots")
        role_slots = role_slots if isinstance(role_slots, int) and role_slots > 0 else None

        # `runtime.cache.slots_by_port` is the AUTHORITY, not the launch entries.
        # An alias is tagged onto only the first N of its host's instances but
        # serves the whole fleet (WP-13), so its entries do not cover every port
        # in `serving.ports` — reading entries alone left the untagged ports on
        # the role-level count and put `-np 4` halves behind a limit of 16, which
        # is the same per-role-fold defect in a new place. `slots_by_port` is
        # built over the inherited fleet and does cover them.
        runtime_cache = stack_prior_launch(record).get("runtime")
        runtime_cache = runtime_cache.get("cache") if isinstance(runtime_cache, dict) else None
        by_port = runtime_cache.get("slots_by_port") if isinstance(runtime_cache, dict) else None

        entry_ports: set[int] = set()
        if isinstance(by_port, dict):
            for raw_port, value in by_port.items():
                try:
                    port = int(raw_port)
                except (TypeError, ValueError):
                    continue
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    continue
                entry_ports.add(port)
                _bump(f"http://localhost:{port}", value)

        for entry in stack_prior_launch_entries(record):
            port = entry.get("port")
            if not isinstance(port, int) or port in entry_ports:
                continue
            entry_slots = entry.get("slots")
            if not isinstance(entry_slots, int) or entry_slots <= 0:
                entry_slots = role_slots
            if entry_slots is None:
                continue
            entry_ports.add(port)
            _bump(f"http://localhost:{port}", entry_slots)

        if role_slots is None:
            continue
        # Ports the record declares but has no launch entry for, plus the
        # endpoint URL itself when it is not a localhost port form. Falls back to
        # the role-level count — this is the pre-2026-08-02 behaviour, retained
        # for exactly the rows that have nothing better.
        for port in stack_prior_serving_ports(serving):
            if port not in entry_ports:
                _bump(f"http://localhost:{port}", role_slots)
        endpoint = serving.get("endpoint")
        if isinstance(endpoint, str):
            endpoint_port = None
            try:
                endpoint_port = stack_prior_endpoint_port(serving)
            except ValueError:
                endpoint_port = None
            if endpoint_port is None or endpoint_port not in entry_ports:
                _bump(endpoint, role_slots)
            else:
                _bump(endpoint, limits.get(f"http://localhost:{endpoint_port}", role_slots))
    return limits


def _stack_prior_serves_llama_slots(serving: dict[str, Any]) -> bool:
    """Return True when a serving block identifies a llama-server slots API."""
    binary = serving.get("binary")
    if binary in {"llama.cpp", "ik-pr1744"}:
        return True
    launch = serving.get("launch")
    runtime = launch.get("runtime") if isinstance(launch, dict) else None
    binary_path = runtime.get("binary_path") if isinstance(runtime, dict) else None
    return isinstance(binary_path, str) and Path(binary_path).name == "llama-server"


def live_stack_slot_query_ports(path: Path = DEFAULT_OUTPUT) -> dict[str, list[int]]:
    """Return live llama-server slot-query ports keyed by canonical role."""
    ports_by_role: dict[str, set[int]] = {}
    for role, record in live_stack_role_records(path).items():
        serving = stack_prior_serving(record)
        if not _stack_prior_serves_llama_slots(serving):
            continue
        for entry in stack_prior_launch_entries(record):
            if entry.get("alias") is True:
                continue
            port = entry.get("port")
            if isinstance(port, int):
                ports_by_role.setdefault(role, set()).add(port)

    return {
        role: sorted(ports)
        for role, ports in sorted(ports_by_role.items())
        if ports
    }


def live_role_primary_ports(
    role_names: set[str] | frozenset[str],
    path: Path = DEFAULT_OUTPUT,
) -> dict[str, int]:
    """Return one primary serving port per requested live role."""
    ports: dict[str, int] = {}
    for role, record in live_stack_role_records(path).items():
        if role not in role_names:
            continue
        serving = stack_prior_serving(record)
        primary_port = stack_prior_primary_port(serving)
        if primary_port is not None:
            ports[role] = primary_port
    return ports


def live_warm_worker_slots(path: Path = DEFAULT_OUTPUT) -> dict[str, int]:
    """Return live warm worker roles and their stack-prior slot caps."""
    caps: dict[str, int] = {}
    for role, record in live_stack_role_records(path).items():
        if not role.startswith("worker_"):
            continue
        serving = stack_prior_serving(record)
        if serving.get("tier") != "warm":
            continue
        slots = serving.get("slots")
        caps[role] = slots if isinstance(slots, int) and slots > 0 else 1
    return caps


def _coerce_tps(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if parsed > 0 else None
    if isinstance(value, str):
        matches = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", value)]
        return max(matches) if matches else None
    return None


def _residency_cost(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    return RESIDENCY_COST.get(value.strip().lower())


def _role_capabilities(
    role: str,
    role_cfg: dict[str, Any] | None,
    registry_roles: dict[str, Any] | None = None,
    stack_aliases: dict[str, Any] | None = None,
) -> list[str]:
    """Capability tags for a role, INHERITED from its host when it declares none.

    Five of the six alias roles carry their own `candidate_roles`; worker_explore
    does not, because it has no `roles:` entry at all — it exists only inside
    `server_mode.worker.shared_with`. Without inheritance it resolved to zero
    capabilities and fell through to the fleet-wide aggregate, silently scoring
    on a different basis from worker_general and worker_math, which run the same
    model on the same server.
    """
    own = []
    if isinstance(role_cfg, dict):
        own = [str(c) for c in (role_cfg.get("candidate_roles") or []) if isinstance(c, str)]
    if own:
        return own

    host = (stack_aliases or {}).get(role)
    if not isinstance(host, str):
        return []
    host_cfg = (registry_roles or {}).get(host)
    if not isinstance(host_cfg, dict):
        return []
    return [str(c) for c in (host_cfg.get("candidate_roles") or []) if isinstance(c, str)]


def _quality_for_role(
    descriptor: dict[str, Any],
    role_cfg: dict[str, Any] | None,
    role: str = "",
    registry_roles: dict[str, Any] | None = None,
    stack_aliases: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve the role-facing quality figure HERE, per role — not per model.

    Capabilities are a property of the ROLE (`roles.<role>.candidate_roles`),
    while a descriptor is per MODEL and is shared by every role bound to it.
    Resolving on the descriptor therefore gave every co-hosted role the same
    axis: architect_general and coder_escalation both ran the 27B (its host is
    architect_critic since the 2026-09-27 ARCHITECT SWAP), so
    coder_escalation inherited `reasoning` when the axis it should be judged
    on is `agentic_coding`. Same weights, different job, different yardstick.
    """
    quality = descriptor.get("quality")
    if not isinstance(quality, dict):
        return None
    public = quality.get("public")
    if not isinstance(public, dict):
        return None
    benchmarks = public.get("benchmarks")
    if not isinstance(benchmarks, dict):
        return None

    capabilities = _role_capabilities(role, role_cfg, registry_roles, stack_aliases)

    from src.registry.model_descriptors import resolve_role_quality

    suite_vector = quality.get("suite_vector")
    overall = suite_vector.get("overall") if isinstance(suite_vector, dict) else None
    return resolve_role_quality(capabilities, benchmarks, overall)


def _quality_axis_values(descriptor: dict[str, Any]) -> dict[str, Any]:
    """Benchmark-keyed quality evidence, with provenance kept separate.

    `local` holds figures measured on THIS stack; `public` holds vendor /
    leaderboard claims about the upstream bf16 model. They are never merged:
    a vendor number and a local number for the same benchmark can differ
    substantially (Qwen3.6-27B SWE-bench Verified: vendor 0.772, local 0.575),
    and collapsing them would destroy exactly the distinction a consumer needs.
    """
    out: dict[str, Any] = {}
    quality = descriptor.get("quality")
    if not isinstance(quality, dict):
        return out
    suite_vector = quality.get("suite_vector")
    if isinstance(suite_vector, dict):
        local = {str(k): v for k, v in suite_vector.items() if k != "overall"}
        if local:
            out["local"] = local
    public = quality.get("public")
    if isinstance(public, dict) and isinstance(public.get("benchmarks"), dict):
        out["public"] = dict(public["benchmarks"])
        out["public_conditions"] = {
            "provenance": public.get("provenance"),
            "reported_precision": public.get("reported_precision"),
            "reported_mode": public.get("reported_mode"),
            "served_precision": public.get("served_precision"),
            "source_url": public.get("source_url"),
            "retrieved": public.get("retrieved"),
        }
        if public.get("universal_keys"):
            out["comparable_across_all_models"] = list(public["universal_keys"])
    return out


def _quality_evidence_axes(descriptor: dict[str, Any]) -> list[str]:
    """Every quality axis this model has ANY evidence on, locally or published.

    Used to tell "measured on domain axes but never whole-suite" (an advisory:
    quality_overall is correctly null) apart from "no quality evidence at all"
    (a real gap). Returns benchmark/suite keys, never an aggregate, because the
    caller must not be tempted to average across them.
    """
    axes: list[str] = []
    quality = descriptor.get("quality")
    if not isinstance(quality, dict):
        return axes
    suite_vector = quality.get("suite_vector")
    if isinstance(suite_vector, dict):
        axes.extend(str(k) for k in suite_vector if k != "overall")
    public = quality.get("public")
    if isinstance(public, dict):
        benchmarks = public.get("benchmarks")
        if isinstance(benchmarks, dict):
            axes.extend(str(k) for k in benchmarks)
    return sorted(dict.fromkeys(axes))


def _quality_prior(descriptor: dict[str, Any]) -> float | None:
    quality = descriptor.get("quality")
    if not isinstance(quality, dict):
        return None
    suite_vector = quality.get("suite_vector")
    if not isinstance(suite_vector, dict):
        return None
    value = suite_vector.get("overall")
    if isinstance(value, (int, float)) and 0 <= float(value) <= 1:
        return float(value)
    return None


def _throughput_prior(descriptor: dict[str, Any], server_cfg: dict[str, Any] | None) -> float | None:
    candidates: list[float] = []
    if isinstance(server_cfg, dict):
        tps = _coerce_tps(server_cfg.get("throughput"))
        if tps is not None:
            candidates.append(tps)

    speed = descriptor.get("speed")
    if isinstance(speed, dict):
        for key in (
            "solo_96t_tps",
            "quarter_48t_tps",
            "prefill_tps",
            "optimized_tps",
            "generation_tps_range",
        ):
            tps = _coerce_tps(speed.get(key))
            if tps is not None:
                candidates.append(tps)
    return max(candidates) if candidates else None


def _descriptor_roles(descriptor: dict[str, Any]) -> list[str]:
    role_bindings = descriptor.get("role_bindings")
    if not isinstance(role_bindings, dict):
        return []
    roles = role_bindings.get("roles")
    if not isinstance(roles, list):
        return []
    return sorted(str(role) for role in roles if isinstance(role, str))


def _descriptor_server_roles(descriptor: dict[str, Any]) -> list[str]:
    role_bindings = descriptor.get("role_bindings")
    if not isinstance(role_bindings, dict):
        return []
    server_roles = role_bindings.get("server_roles")
    if not isinstance(server_roles, list):
        return []
    return sorted(str(role) for role in server_roles if isinstance(role, str))


def _descriptor_by_role(descriptors: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_role: dict[str, dict[str, Any]] = {}
    models = descriptors.get("models")
    if not isinstance(models, list):
        return by_role
    for descriptor in models:
        if not isinstance(descriptor, dict):
            continue
        for role in _descriptor_roles(descriptor):
            by_role[role] = descriptor
    return by_role


def _server_for_role(
    role: str,
    server_mode: dict[str, Any],
    stack_aliases: dict[str, str],
    stack_roles: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None, str]:
    primary = stack_aliases.get(role)
    direct = server_mode.get(role)
    if isinstance(direct, dict):
        # A logical alias can retain its own registry row for routing metadata.
        # When that row omits a launch-affecting host field, however, project
        # the manifest primary's process contract rather than publishing a
        # partial runtime record for the same llama-server.
        if primary:
            server_role, host_cfg, binding = _server_for_role(
                primary, server_mode, {}, stack_roles
            )
            if (
                host_cfg is not None
                and direct.get("model_role") == host_cfg.get("model_role")
                and not direct.get("draft_model")
                and host_cfg.get("draft_model")
            ):
                return server_role, host_cfg, f"stack_manifest.alias->{binding}"
        return role, direct, "server_mode.direct"

    for server_role, cfg in server_mode.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("model_role") == role:
            return str(server_role), cfg, "server_mode.model_role"
        shared_with = cfg.get("shared_with")
        if isinstance(shared_with, list) and role in shared_with:
            return str(server_role), cfg, "server_mode.shared_with"

    if primary:
        server_role, cfg, binding = _server_for_role(primary, server_mode, {}, stack_roles)
        if cfg is not None:
            return server_role, cfg, f"stack_manifest.alias->{binding}"

    stack_cfg = stack_roles.get(role)
    if isinstance(stack_cfg, dict):
        return role, stack_cfg, "stack_manifest.role"

    return None, None, "unresolved"


def _launch_mode_for_server(server: dict[str, Any]) -> str:
    if server.get("worker_pool"):
        return "worker_pool"
    if server.get("vision"):
        return "vision"
    if server.get("embedding"):
        return "embedding"
    return "default"


def _launch_entry_for_role(
    server: dict[str, Any],
    role: str,
    shape_class: str | None = None,
) -> dict[str, Any] | None:
    port = server.get("port")
    roles = server.get("roles")
    if not isinstance(port, int) or not isinstance(roles, list) or not roles:
        return None
    primary_role = roles[0] if isinstance(roles[0], str) else role
    entry: dict[str, Any] = {
        "port": port,
        "primary_role": primary_role,
        "mode": _launch_mode_for_server(server),
        "alias": role != primary_role,
    }
    numa_instance = server.get("numa_instance")
    if isinstance(numa_instance, int):
        entry["numa_instance"] = numa_instance
    # 2026-08-02: the instance's SHAPE CLASS (full/half/quarter/gpu_host_lane),
    # resolved from the `cpu_shape` its stack_topology.yaml entry declares. This
    # is the launcher-side half of the per-instance `-np` join and is carried on
    # the entry so `_launch_runtime_record` can multiply it against the master
    # registry it was HANDED, rather than against the ambient one.
    if isinstance(shape_class, str) and shape_class:
        entry["cpu_shape_class"] = shape_class
    worker_type = server.get("worker_type")
    if isinstance(worker_type, str):
        entry["worker_type"] = worker_type
    vision_type = server.get("vision_type")
    if isinstance(vision_type, str):
        entry["vision_type"] = vision_type
    return entry


def _launch_requirements_for_meta(
    meta: dict[str, Any],
    *,
    worker_pool_models: dict[str, Any],
    explore_draft_model: Any,
    vision_worker_model: Any,
    vision_worker_mmproj: Any,
    vision_escalation_model: Any,
    vision_escalation_mmproj: Any,
) -> dict[str, str]:
    requirements: dict[str, str] = {}
    mode = str(meta.get("mode") or "")
    if mode == "worker_pool":
        worker_type = str(meta.get("worker_type") or "")
        model_path = worker_pool_models.get(worker_type)
        if model_path:
            requirements["model_path"] = str(model_path)
        if worker_type == "explore" and explore_draft_model:
            requirements["draft_model_path"] = str(explore_draft_model)
    elif mode == "vision":
        vision_type = meta.get("vision_type")
        if vision_type == "worker":
            requirements["model_path"] = str(vision_worker_model)
            requirements["mmproj_path"] = str(vision_worker_mmproj)
        elif vision_type == "escalation":
            requirements["model_path"] = str(vision_escalation_model)
            requirements["mmproj_path"] = str(vision_escalation_mmproj)
    return {key: value for key, value in sorted(requirements.items()) if value}


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _effective_context_for_meta(
    role: str,
    meta: dict[str, Any],
    *,
    kv_context_sizes: dict[str, Any],
    default_context_size: Any,
) -> int | None:
    return _positive_int(kv_context_sizes.get(role, default_context_size))


def _launch_record(
    entries: list[dict[str, Any]],
    requirements: dict[str, str] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sorted_entries = sorted(
        entries,
        key=lambda entry: (
            entry.get("port", -1),
            str(entry.get("primary_role", "")),
            str(entry.get("mode", "")),
        ),
    )
    return {
        "entries": sorted_entries,
        "primary_roles": sorted(
            {
                str(entry["primary_role"])
                for entry in sorted_entries
                if isinstance(entry.get("primary_role"), str)
            }
        ),
        "modes": sorted(
            {str(entry["mode"]) for entry in sorted_entries if isinstance(entry.get("mode"), str)}
        ),
        "requirements": copy.deepcopy(requirements or {}),
        "runtime": copy.deepcopy(runtime or {}),
    }


class StackPriorsModeError(RuntimeError):
    """A priors compile could not resolve a realized NUMA mode and refuses to
    silently default to ``full`` (ESC-8 Fix 6 / kill chain A4).

    Raised only on the WRITE/check compile path (``require_realized_mode``); a
    plain ``compile_stack_priors()`` keeps the legacy launcher-default behavior.
    """


def _realized_compile_numa_mode(
    *,
    environ: Mapping[str, str] | None = None,
    connect: Callable[[str, int], bool] | None = None,
) -> str:
    """Resolve the NUMA mode for a WRITE-path priors compile against the live fleet.

    ESC-8 Fix 6. A clean-shell ``stack_change_pipeline.py update`` must never
    read the ambient default-``full`` env and rewrite stack_priors.yaml to the
    dead full lineup (kill chain A4). The mode is derived from the *realized*
    fleet (a bare TCP probe of the quarterable-role ports, via
    ``realized_fleet``; ``connect`` is injectable so tests never touch sockets):

      * env unset  -> use the realized mode.
      * env set, agrees with realized (or realized is ``both``) -> use env.
      * env set, contradicts realized -> prefer realized, log the correction.
      * no realized signal (nothing listening / probe unavailable) -> REFUSE
        (raise ``StackPriorsModeError``) with an explicit-mode instruction —
        never silently default to ``full``.
    """
    from scripts.server.stack_numa_mode import normalize_stack_numa_mode

    env = os.environ if environ is None else environ
    raw = env.get("ORCHESTRATOR_STACK_NUMA_MODE")
    env_mode = normalize_stack_numa_mode(raw) if raw else None

    from scripts.server.realized_fleet import derive_realized_numa_mode

    realized = derive_realized_numa_mode(connect=connect)

    if realized is None:
        raise StackPriorsModeError(
            "refusing to compile stack priors: no quarterable-role port is "
            "listening, so the realized NUMA mode cannot be determined (stack "
            "down or probe unavailable). Not defaulting to 'full' — ESC-8 kill "
            "chain A4 would rewrite stack_priors.yaml to the dead full lineup. "
            "Re-run with an explicit ORCHESTRATOR_STACK_NUMA_MODE once the "
            "intended fleet is up (e.g. ORCHESTRATOR_STACK_NUMA_MODE=quarter)."
        )

    if env_mode is None or realized == "both" or env_mode == realized:
        return env_mode or realized
    _LOGGER.warning(
        "ORCHESTRATOR_STACK_NUMA_MODE=%s contradicts the realized fleet mode %r; "
        "preferring the realized mode for the stack-priors compile (ESC-8 Fix 6).",
        env_mode,
        realized,
    )
    return realized


def _stack_manifest_info(
    numa_mode: str | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    try:
        from scripts.server.stack_manifest import (
            EXPLORE_DRAFT_MODEL,
            DEFAULT_EFFECTIVE_CONTEXT_TOKENS,
            HOT_SERVERS,
            LAUNCH_CONTEXT_TOKENS,
            PORT_MAP,
            ROLE_LAUNCH_META,
            VISION_ESCALATION_MMPROJ,
            VISION_ESCALATION_MODEL,
            VISION_WORKER_MMPROJ,
            VISION_WORKER_MODEL,
            WARM_SERVERS,
            WORKER_POOL_MODELS,
            _filter_by_numa_mode,
        )
        from scripts.server.stack_numa import instance_shape_class
    except Exception:
        return {}, {}

    # ESC-8 Fix 6: an explicit ``numa_mode`` (resolved against the realized fleet
    # by the WRITE/check compile path) wins; when None, preserve the legacy
    # launcher-default behavior (env, default full) so direct callers and the
    # existing reader-agreement tests are unaffected.
    from scripts.server.stack_numa_mode import env_stack_numa_mode, normalize_stack_numa_mode

    if numa_mode is None:
        numa_mode = env_stack_numa_mode()
    else:
        numa_mode = normalize_stack_numa_mode(numa_mode)
    active_servers = _filter_by_numa_mode(HOT_SERVERS + WARM_SERVERS, numa_mode)

    launch_ports_by_role: dict[str, list[int]] = {}
    launch_entries_by_role: dict[str, list[dict[str, Any]]] = {}
    # port -> shape class over the ACTIVE (numa-mode-filtered) fleet. Kept next to
    # the entry lists rather than derived per role because an ALIAS inherits its
    # host's whole port fleet (WP-13 fleet convergence) while being TAGGED onto
    # only the first N instances — so its own entries cannot describe every port
    # it serves, and resolving `-np` from entries alone left the untagged ports on
    # the role-level fallback. That produced `toolrunner` claiming 16 slots on
    # :8182, a half instance that runs 4.
    shape_class_by_port: dict[int, str] = {}
    for server in active_servers:
        if not isinstance(server, dict):
            continue
        port = server.get("port")
        if not isinstance(port, int):
            continue
        server_roles = server.get("roles") or []
        host_role = server_roles[0] if server_roles and isinstance(server_roles[0], str) else None
        # Shape class is a property of the INSTANCE, so it is looked up on the
        # host role (which owns the topology entry) and inherited by every alias
        # riding the same process — an alias has no placement of its own.
        shape_class = (
            instance_shape_class(host_role, server.get("numa_instance", 0) or 0)
            if isinstance(host_role, str)
            else None
        )
        if isinstance(shape_class, str) and shape_class:
            shape_class_by_port[port] = shape_class
        for role in server_roles:
            if isinstance(role, str):
                launch_ports_by_role.setdefault(role, []).append(port)
                launch_entry = _launch_entry_for_role(server, role, shape_class)
                if launch_entry is not None:
                    launch_entries_by_role.setdefault(role, []).append(launch_entry)

    aliases: dict[str, str] = {}
    roles: dict[str, dict[str, Any]] = {}
    for primary, meta in ROLE_LAUNCH_META.items():
        if not isinstance(meta, dict):
            continue
        launch_requirements = _launch_requirements_for_meta(
            meta,
            worker_pool_models=WORKER_POOL_MODELS,
            explore_draft_model=EXPLORE_DRAFT_MODEL,
            vision_worker_model=VISION_WORKER_MODEL,
            vision_worker_mmproj=VISION_WORKER_MMPROJ,
            vision_escalation_model=VISION_ESCALATION_MODEL,
            vision_escalation_mmproj=VISION_ESCALATION_MMPROJ,
        )
        ports = sorted(set(launch_ports_by_role.get(str(primary), [])))
        if ports:
            port = ports[0]
        elif meta.get("no_numa"):
            port = meta.get("port")
        else:
            port = PORT_MAP.get(primary)
        primary_ports = ports or ([port] if isinstance(port, int) else [])
        roles[str(primary)] = {
            "tier": meta.get("tier"),
            "port": port,
            "ports": primary_ports,
            "port_shape_classes": {
                p: shape_class_by_port[p] for p in primary_ports if p in shape_class_by_port
            },
            "url": f"http://localhost:{port}" if isinstance(port, int) else None,
            "effective_context_tokens": _effective_context_for_meta(
                str(primary),
                meta,
                kv_context_sizes=LAUNCH_CONTEXT_TOKENS,
                default_context_size=DEFAULT_EFFECTIVE_CONTEXT_TOKENS,
            ),
            "launch": _launch_record(
                launch_entries_by_role.get(str(primary), []),
                launch_requirements,
            ),
        }
        # gpu-serving-tie-in P2-6 (P0-1): a launcher-only entry may name a
        # registry TENANT role (``tenant_role``). Synthesize a stack-role record
        # for it so the priors compile can emit a resolvable launch record —
        # marked ``launcher_only_tenant`` so ``_role_record`` classifies it
        # ``launcher_tenant`` (NEVER ``live_stack``). Inert today: no production
        # launch-meta entry carries ``tenant_role``.
        if meta.get("launcher_only") is True:
            tenant = meta.get("tenant_role")
            if isinstance(tenant, str) and tenant and tenant not in roles:
                tenant_mode = str(meta.get("mode") or "default")
                shape = (
                    _gpu_shadow_lane_serving_shape()
                    if tenant_mode == "gpu_shadow_lane"
                    else None
                )
                tenant_entries = (
                    [
                        {
                            "port": port,
                            "primary_role": str(primary),
                            "mode": tenant_mode,
                            "alias": False,
                        }
                    ]
                    if isinstance(port, int)
                    else []
                )
                roles[tenant] = {
                    "tier": meta.get("tier"),
                    "port": port,
                    "ports": [port] if isinstance(port, int) else [],
                    "url": f"http://localhost:{port}" if isinstance(port, int) else None,
                    "effective_context_tokens": (
                        shape.get("context_tokens") if isinstance(shape, dict) else None
                    ),
                    "launcher_only_tenant": True,
                    "launcher_role": str(primary),
                    "launch": _launch_record(tenant_entries),
                }
        shared = meta.get("shared_with_first_n") if isinstance(meta, dict) else None
        if isinstance(shared, list):
            # WP-13 fleet convergence: an alias role (shared_with_first_n) is not a
            # process of its own — it rides the primary/host's llama-server(s). It
            # must therefore inherit the host's FULL serving fleet, not just the
            # subset of instances it was tagged onto (shared_with_first_n_count).
            # Emitting only the tagged ports made the generated serving.ports a
            # single quarter (e.g. worker_math -> [8072, 8082]), diverging from the
            # operative-layer Fix-A default (_server_url_default(host)) which serves
            # the full `full:` quarter fleet and applies the host's region locks +
            # demotion. That divergence serialized worker_math eval traffic on one
            # quarter (EV-11c arm-2). Inherit roles[primary]["ports"] so a future
            # regeneration is byte-identical to the delegated operative URL; fall
            # back to the alias's own launch ports only when the host has no
            # resolved port fleet.
            host_ports = roles[str(primary)].get("ports")
            for alias in shared:
                if isinstance(alias, str):
                    if isinstance(host_ports, list) and host_ports:
                        alias_ports = list(host_ports)
                    else:
                        alias_ports = sorted(set(launch_ports_by_role.get(alias, [])))
                    alias_port = alias_ports[0] if alias_ports else port
                    aliases[alias] = str(primary)
                    resolved_alias_ports = alias_ports or (
                        [alias_port] if isinstance(alias_port, int) else []
                    )
                    roles[alias] = {
                        "tier": meta.get("tier"),
                        "port": alias_port,
                        "ports": resolved_alias_ports,
                        # Inherited over the HOST'S WHOLE FLEET, matching `ports`
                        # above — an alias rides every one of its host's
                        # instances, not only the ones it is tagged onto.
                        "port_shape_classes": {
                            p: shape_class_by_port[p]
                            for p in resolved_alias_ports
                            if p in shape_class_by_port
                        },
                        "url": f"http://localhost:{alias_port}" if isinstance(alias_port, int) else None,
                        "effective_context_tokens": roles[str(primary)].get(
                            "effective_context_tokens"
                        ),
                        "launch": _launch_record(
                            launch_entries_by_role.get(alias, []),
                            launch_requirements,
                        ),
                    }
    return aliases, roles


def _gpu_shadow_lane_serving_shape() -> dict[str, int] | None:
    """Resolve the GPU shadow lane's serving shape from POLICY DATA.

    gpu-serving-tie-in P2-6 (P0-1c): slots/-np and context for a
    ``gpu_shadow_lane``-mode launcher tenant come from the ``serving_shape``
    block of ``orchestration/gpu_shadow_lane_np_ceiling.yaml`` — NEVER from the
    CPU-mode launcher defaults (SERIAL_ROLES 2-slot, 32768-token context),
    which are wrong for the lane. Returns None when the shape is missing or
    invalid; callers surface that as a known_gap / null slots (refuse — the
    pipeline gates catch it), never by guessing a shape.
    """
    try:
        from scripts.server.gpu_shadow_lane import load_serving_shape

        return load_serving_shape()
    except Exception:
        return None


def _first_string(values: Any) -> str | None:
    if not isinstance(values, list):
        return None
    for value in values:
        if isinstance(value, str):
            return value
    return None


def _first_launch_entry_value(launch: dict[str, Any], field: str) -> str | None:
    entries = launch.get("entries")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get(field), str):
            return str(entry[field])
    return None


def _runtime_requirements(
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
) -> tuple[str | None, list[str]]:
    runtime = server_cfg.get("runtime_requirements") if isinstance(server_cfg, dict) else None
    if not isinstance(runtime, dict) and isinstance(role_cfg, dict):
        role_server = role_cfg.get("server")
        if isinstance(role_server, dict):
            runtime = role_server.get("runtime_requirements")
    if not isinstance(runtime, dict):
        return None, []
    binary_dir = runtime.get("binary_dir") if isinstance(runtime.get("binary_dir"), str) else None
    raw_ld = runtime.get("ld_library_path")
    ld_paths = [str(path) for path in raw_ld if isinstance(path, str)] if isinstance(raw_ld, list) else []
    return binary_dir, ld_paths


def _declared_device(server_cfg: Any, role_cfg: Any) -> str | None:
    """Return the role's declared serving device, if any (e.g. 'ROCm0', 'none')."""
    for cfg in (server_cfg, role_cfg):
        if isinstance(cfg, dict):
            device = cfg.get("device")
            if isinstance(device, str) and device.strip():
                return device.strip()
    return None


def _backend_for_role(server_cfg: Any, role_cfg: Any) -> str:
    """Map a role's declared device to a production kernel BACKEND.

    A backend is a capability, not a location. `llama.cpp/build/bin` is a CPU-ONLY
    build with no `libggml-hip.so`, so a role moved to the GPU by a registry edit
    alone would previously have launched on it and run on CPU SILENTLY — a missing
    ggml backend does not raise, it simply is not used. Deriving the backend from
    the declared device is what makes `device: ROCm0` mean something at launch.
    """
    device = (_declared_device(server_cfg, role_cfg) or "").lower()
    if device.startswith("rocm") or device.startswith("cuda") or device.startswith("gpu"):
        return "gpu"
    return "cpu"


def _effective_acceleration(
    role_cfg: dict[str, Any] | None,
    server_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(server_cfg, dict) and isinstance(server_cfg.get("acceleration"), dict):
        return copy.deepcopy(server_cfg["acceleration"])
    if isinstance(role_cfg, dict) and isinstance(role_cfg.get("acceleration"), dict):
        return copy.deepcopy(role_cfg["acceleration"])
    return {}


def _override_kv_args(acceleration: dict[str, Any]) -> list[str]:
    if acceleration.get("type") != "moe_expert_reduction":
        return []
    override_key = acceleration.get("override_key")
    experts = acceleration.get("experts")
    if not isinstance(override_key, str) or not isinstance(experts, int):
        return []
    return [f"{override_key}=int:{experts}"]


def _role_no_mmap_prior(
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
    *,
    default: bool,
) -> bool:
    """Resolve a role's no_mmap cache prior from its config.

    2026-06-26 v6 cutover: precedence is server_mode -> roles block -> ``default``.
    A role may set ``no_mmap: true`` directly, or under a ``cache``/``serving``
    sub-mapping. Absent any explicit setting the caller-supplied ``default`` is
    used (which preserves the legacy worker_pool+explore canonical-recipe value).
    """
    for cfg in (server_cfg, role_cfg):
        if not isinstance(cfg, dict):
            continue
        if isinstance(cfg.get("no_mmap"), bool):
            return cfg["no_mmap"]
        for nested_key in ("cache", "serving"):
            nested = cfg.get(nested_key)
            if isinstance(nested, dict) and isinstance(nested.get("no_mmap"), bool):
                return nested["no_mmap"]
    return default


def _spec_type_has_mtp(spec_type: str | None) -> bool:
    """True when a registry ``spec_type`` engages the MTP/NEXTN self-draft path.

    ``spec_type`` may be a single token (``draft-mtp``) or a COMPOSED,
    comma-separated chain (``ngram-mod,draft-mtp`` — the production recipe; see
    the canonical ``speculative_decoding_policy`` block in the master registry).
    A composed chain still needs the NEXTN draft path resolved and ``-md``
    suppression applied, so it must take the same branch as the bare token.

    Testing ``== "draft-mtp"`` here (the pre-2026-07-31 behaviour) made a
    composed value fall through to the DISABLED spec, launching the role with no
    speculation at all.
    """
    if not isinstance(spec_type, str) or not spec_type:
        return False
    return "draft-mtp" in {token.strip() for token in spec_type.split(",")}


# DRAFT-SEL-1: drafter spec types that launch with a DRAFT MODEL (`-md`), as
# opposed to the NEXTN self-draft above. Before this, any spec_type without the
# `draft-mtp` token fell through to the DISABLED spec, so selecting DFlash2 in the
# registry would have launched :8083 with no speculation at all — the same
# fall-through class the 2026-07-31 composed-recipe fix closed for `ngram-mod,draft-mtp`.
_EXTERNAL_DRAFTER_SPEC_TYPES = frozenset({"draft-dflash", "draft-simple", "draft-eagle3"})


def _spec_type_launches_drafter(spec_type: str | None) -> bool:
    """True when ``spec_type`` needs a resolved drafter path and an enabled spec."""
    if _spec_type_has_mtp(spec_type):
        return True
    if not isinstance(spec_type, str) or not spec_type:
        return False
    return bool(
        {token.strip() for token in spec_type.split(",")} & _EXTERNAL_DRAFTER_SPEC_TYPES
    )


def _positive_int_prior(
    *containers: dict[str, Any] | None,
    key: str,
    fallback: int,
) -> int:
    for container in containers:
        if not isinstance(container, dict):
            continue
        value = container.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return fallback


def _nonnegative_int_prior(
    *containers: dict[str, Any] | None,
    key: str,
) -> int | None:
    for container in containers:
        if not isinstance(container, dict):
            continue
        value = container.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _runtime_flag_string_prior(
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
    *,
    key: str,
    fallback: str | None = None,
) -> str | None:
    """Resolve an optional launcher flag from registry serving metadata.

    Dedicated stack-manifest roles such as vision_escalation do not have a
    server_mode row, so role-local ``server``/``serving`` metadata must be able
    to carry runtime flags into the generated stack priors.
    """
    for cfg in (server_cfg, role_cfg):
        if not isinstance(cfg, dict):
            continue
        # `serving_shape` FIRST: it is the master registry's grouped
        # KV-feasibility declaration (n_ctx / slots_by_shape / kv_quant), and the
        # group must outrank any surviving flat copy of one of its members.
        for source in (
            cfg.get("serving_shape"),
            cfg,
            cfg.get("server"),
            cfg.get("serving"),
            cfg.get("launch"),
        ):
            if isinstance(source, dict):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return value
    return fallback


def _runtime_flag_int_prior(
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
    *,
    key: str,
    minimum: int = 0,
) -> int | None:
    """Resolve an optional INTEGER launcher flag from registry serving metadata.

    The integer twin of ``_runtime_flag_string_prior``, with the same search order
    (``server_mode`` row first, then role-local ``server`` / ``serving`` / ``launch``
    sub-mappings) so a role whose declaration lives only under ``roles.<role>.serving``
    — which is where ``worker_vision`` declares ``n_gpu_layers``, ``image_min_tokens``
    and ``cache_ram`` — can still carry it into the compiled priors.

    ``minimum=0`` is deliberate and load-bearing: ``cache_ram: 0`` ("prompt cache
    disabled") is a DECLARED value, not an absent one, and a ``> 0`` test would drop
    it. Returns ``None`` when undeclared. Never substitute a default here — these are
    GPU-shaped flags, and a default would put ``-ngl`` on every CPU role.
    """
    for cfg in (server_cfg, role_cfg):
        if not isinstance(cfg, dict):
            continue
        # `serving_shape` FIRST: it is the master registry's grouped
        # KV-feasibility declaration (n_ctx / slots_by_shape / kv_quant), and the
        # group must outrank any surviving flat copy of one of its members.
        for source in (
            cfg.get("serving_shape"),
            cfg,
            cfg.get("server"),
            cfg.get("serving"),
            cfg.get("launch"),
        ):
            if not isinstance(source, dict):
                continue
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= minimum:
                return value
    return None


def _runtime_flag_bool_prior(
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
    *,
    key: str,
) -> bool | None:
    """Resolve an optional BOOLEAN launcher fact from registry serving metadata.

    The boolean twin of ``_runtime_flag_int_prior``, with the same search order
    (``serving_shape`` first, then the row itself, then the role-local ``server`` /
    ``serving`` / ``launch`` sub-mappings). ``False`` is a DECLARED value, not an
    absent one: it compiles to an explicit negative flag, so a fact the operator
    pinned never silently reverts to whatever the kernel's default happens to be.
    Returns ``None`` when undeclared, and the launcher then emits nothing.

    Origin (2026-09-24, stack-change-kvu): llama-server enables a unified KV pool
    ONLY when ``-np`` is left on auto (``tools/server/server.cpp:145-150``); the
    launcher always passes ``-np``, so ``:8083`` ran split KV while the master
    registry asserted "KV IS UNIFIED". A default is not a declaration.
    """
    for cfg in (server_cfg, role_cfg):
        if not isinstance(cfg, dict):
            continue
        for source in (
            cfg.get("serving_shape"),
            cfg,
            cfg.get("server"),
            cfg.get("serving"),
            cfg.get("launch"),
        ):
            if not isinstance(source, dict):
                continue
            value = source.get(key)
            if isinstance(value, bool):
                return value
    return None


# Registry spellings for GPU offload depth, in precedence order. Two are in live
# use with different value types; see _n_gpu_layers_prior.
_N_GPU_LAYERS_KEYS: tuple[str, ...] = ("n_gpu_layers", "ngl", "gpu_layers")


def _n_gpu_layers_prior(
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
) -> int | str | None:
    """Resolve the declared GPU offload depth, whatever the registry chose to call it.

    The master registry declares this TWICE, under two spellings and two types:
    ``roles.worker_vision.serving.n_gpu_layers: 999`` (int) and
    ``server_mode.architect_general.ngl: "all"`` (str; the 27B's row, which is
    ``server_mode.architect_critic`` since the 2026-09-27 ARCHITECT SWAP). Reading
    only the first spelling is how that role would end up emitting ``--device ROCm0`` with
    no ``-ngl`` at all — a GPU launch in name only, since the device is then selected
    and nothing is offloaded to it. ``-ngl all`` is a valid llama-server value on
    this kernel (the GPU shadow-lane builder already emits it), so the string form is
    carried through verbatim rather than being translated to a number here.

    Returns None when undeclared. No default: a role that does not ask for offload
    must not receive it.
    """
    for key in _N_GPU_LAYERS_KEYS:
        for cfg in (server_cfg, role_cfg):
            if not isinstance(cfg, dict):
                continue
            # `serving_shape` FIRST: it is the master registry's grouped
            # KV-feasibility declaration (n_ctx / slots_by_shape / kv_quant), and
            # the group must outrank any surviving flat copy of one of its members.
            for source in (
                cfg.get("serving_shape"),
                cfg,
                cfg.get("server"),
                cfg.get("serving"),
                cfg.get("launch"),
            ):
                if not isinstance(source, dict):
                    continue
                value = source.get(key)
                if isinstance(value, bool):
                    continue
                if isinstance(value, int) and value >= 0:
                    return value
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _descriptor_reasoning_flag_prior(descriptor: dict[str, Any]) -> str | None:
    acceleration = descriptor.get("acceleration")
    if not isinstance(acceleration, dict):
        return None
    if acceleration.get("enable_thinking") is not False:
        return None
    thinking_control = acceleration.get("thinking_control")
    mode = thinking_control.get("mode") if isinstance(thinking_control, dict) else None
    if mode in {"reasoning_off", "toggle_off"}:
        return "off"
    return None


def _runtime_reasoning_prior(
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
    descriptor: dict[str, Any],
    *,
    mode: str,
    worker_type: str | None,
    vision_type: str | None,
    vision_escalation_reasoning: str,
) -> str | None:
    explicit = _runtime_flag_string_prior(server_cfg, role_cfg, key="reasoning")
    if explicit:
        return explicit
    if mode == "vision" and vision_type == "escalation":
        return vision_escalation_reasoning
    if mode == "worker_pool" and worker_type == "explore":
        return "off"
    return _descriptor_reasoning_flag_prior(descriptor)


def _number_prior(
    *containers: dict[str, Any] | None,
    key: str,
    fallback: int | float,
) -> int | float:
    for container in containers:
        if not isinstance(container, dict):
            continue
        value = container.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return fallback


def _nested_mapping(container: dict[str, Any] | None, *path: str) -> dict[str, Any] | None:
    current: Any = container
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, dict) else None


#: llama.cpp's own n_batch default (common/common.h), i.e. the logical batch whenever
#: the launcher emits no ``-b``. llama.cpp clamps ``n_ubatch = min(n_batch, n_ubatch)``.
LLAMACPP_DEFAULT_N_BATCH = 2048

#: serving_shape key naming the ``-ub`` that ``vram_non_kv_gib`` was derived at (optional).
#: The KQ mask ([n_kv, n_ubatch, 1, 1] f16 on a unified pool) lives in that scalar and
#: scales with ``-ub``, so the figure is a function of the micro-batch.
VRAM_UBATCH_KEY = "vram_non_kv_ubatch"


def _declared_batch_shape_prior(
    role: str,
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
    *,
    default_ubatch: int,
) -> tuple[int, int | None]:
    """Resolve a DECLARED ``-ub`` / ``-b`` pair for a default-mode server, or refuse.

    STACKCHG-8083BATCH-20261004. ``serving_shape.ubatch`` and ``serving_shape.batch``
    (same search order as every other serving fact) are optional; undeclared, the
    server keeps ``default_ubatch`` and no ``-b`` (llama.cpp's 2048).

    Two refusals, both at compile time rather than at launch:

    * K4: a declared ``ubatch`` above the effective batch is silently clamped by
      llama.cpp (``n_ubatch = min(n_batch, n_ubatch)``), so the declaration would be
      a promise the launch cannot keep.
    * RESTATED DERIVATION: ``serving_shape.vram_non_kv_gib`` carries the KQ-mask
      term, which scales with ``-ub``. When ``vram_non_kv_ubatch`` names the ``-ub``
      the figure was derived at, the resolved ``-ub`` must equal it; changing one
      without re-deriving the other would pass the capacity gate on a stale number.
    """
    declared_ubatch = _runtime_flag_int_prior(server_cfg, role_cfg, key="ubatch", minimum=1)
    declared_batch = _runtime_flag_int_prior(server_cfg, role_cfg, key="batch", minimum=1)
    ubatch = declared_ubatch if declared_ubatch is not None else default_ubatch
    effective_batch = declared_batch if declared_batch is not None else LLAMACPP_DEFAULT_N_BATCH
    if (declared_ubatch is not None or declared_batch is not None) and ubatch > effective_batch:
        raise ValueError(
            f"role {role!r}: declared -ub {ubatch} exceeds the effective -b {effective_batch} "
            "(llama.cpp clamps n_ubatch = min(n_batch, n_ubatch)); the declared ubatch would be "
            "INERT (K4). Declare serving_shape.batch >= ubatch, or lower ubatch."
        )
    for cfg in (server_cfg, role_cfg):
        shape = _nested_mapping(cfg, "serving_shape")
        derived_for = shape.get(VRAM_UBATCH_KEY) if shape else None
        if derived_for is None:
            continue
        if isinstance(derived_for, bool) or not isinstance(derived_for, int) or derived_for != ubatch:
            raise ValueError(
                f"role {role!r}: serving_shape.vram_non_kv_gib was derived at -ub "
                f"{derived_for!r} ({VRAM_UBATCH_KEY}), but the server resolves -ub {ubatch}. "
                f"Re-derive vram_non_kv_gib (the KQ-mask term scales with -ub) and update "
                f"{VRAM_UBATCH_KEY} in the same change."
            )
        break
    return ubatch, declared_batch


def _kv_types_prior(
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
    descriptor: dict[str, Any],
) -> tuple[str, str] | None:
    """Resolve runtime KV cache types from registry/descriptor facts.

    Registry-derived facts take precedence over launcher fallback tables. The
    accepted shapes mirror the current master registry (`server_mode.*.kv_quant`
    and `roles.*.model.kv_cache`) plus descriptor acceleration metadata for
    future generated surfaces.
    """
    candidate_maps = (
        # The grouped declaration outranks every flat copy — see the
        # `serving_shape:` block comment in the master registry.
        _nested_mapping(server_cfg, "serving_shape", "kv_quant"),
        _nested_mapping(role_cfg, "serving_shape", "kv_quant"),
        _nested_mapping(server_cfg, "kv_quant"),
        _nested_mapping(server_cfg, "kv_cache"),
        _nested_mapping(role_cfg, "model", "kv_cache"),
        _nested_mapping(role_cfg, "kv_cache"),
        _nested_mapping(descriptor, "acceleration", "kv"),
    )
    for candidate in candidate_maps:
        if not isinstance(candidate, dict):
            continue
        key_type = candidate.get("k") or candidate.get("type_k") or candidate.get("kv_type_k")
        value_type = candidate.get("v") or candidate.get("type_v") or candidate.get("kv_type_v")
        if isinstance(key_type, str) and isinstance(value_type, str):
            return key_type, value_type
    return None


def _draft_kv_types_prior(
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
) -> tuple[str, str] | None:
    """Resolve the speculative draft context's KV types (``-ctkd``/``-ctvd``).

    Only ``serving_shape.draft_kv_quant`` is read: this is a GPU-capacity lever
    (SSU-F3 fix #1), not a legacy flat key with history to honour.
    """
    for cfg in (server_cfg, role_cfg):
        candidate = _nested_mapping(cfg, "serving_shape", "draft_kv_quant")
        if isinstance(candidate, dict):
            k, v = candidate.get("k"), candidate.get("v")
            if isinstance(k, str) and k and isinstance(v, str) and v:
                return k, v
    return None


def _worker_context_prior(
    role_cfg: dict[str, Any] | None,
    *,
    fallback: int,
) -> int:
    model_cfg = _nested_mapping(role_cfg, "model")
    if not isinstance(model_cfg, dict):
        return fallback
    return _positive_int_prior(
        model_cfg,
        key="max_context",
        fallback=fallback,
    )


def _resolve_nextn_draft_path(
    requirements: dict[str, Any],
    acceleration: dict[str, Any],
    server_cfg: dict[str, Any] | None,
    *,
    models_dir: Any = None,
) -> str | None:
    """Resolve the NEXTN self-draft GGUF path for a draft-mtp role.

    2026-06-26 v6 cutover: NEXTN self-draft roles (frontdoor, architect_general)
    embed the draft head in the base GGUF. The compiled draft path intentionally
    resolves to the model path; the launcher emits draft-mtp spec flags but
    suppresses ``-md`` when both paths have the same realpath.

    Sources, in precedence order:
      1. requirements.draft_model_path (explicit full path, e.g. server_mode override)
      2. acceleration.draft_model_path (explicit full path on the accel block)
      3. acceleration.draft_model / server_cfg.draft_model (bare or relative; the
         registry's NEXTN self-draft pointer == the base file)
      4. requirements.model_path / server_cfg.model_path (full base path; self-draft)
      5. server_cfg.model (bare or relative base path)
    Bare/relative values are resolved against ``models_dir`` when available so the
    emitted path is absolute.
    """
    candidates = [
        requirements.get("draft_model_path"),
        acceleration.get("draft_model_path"),
        acceleration.get("draft_model"),
        server_cfg.get("draft_model") if isinstance(server_cfg, dict) else None,
        requirements.get("model_path"),
        server_cfg.get("model_path") if isinstance(server_cfg, dict) else None,
        server_cfg.get("model") if isinstance(server_cfg, dict) else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_absolute():
            return str(candidate_path)
        if models_dir is not None:
            return str(Path(models_dir) / candidate)
        return candidate
    return None


def _slots_by_shape_declaration(
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
) -> dict[str, int]:
    """`serving_shape.slots_by_shape` from the registry the COMPILER WAS HANDED.

    Deliberately NOT `stack_manifest.slots_by_shape_for`, which reads the ambient
    master. The compiler takes a `registry_path` and must honour that one; a
    second, ambient read would give one field two sources — the bug one level up.
    """
    out: dict[str, int] = {}
    for cfg in (server_cfg, role_cfg):
        table = _nested_mapping(cfg, "serving_shape", "slots_by_shape")
        if not isinstance(table, dict):
            continue
        for shape_class, value in table.items():
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                out.setdefault(str(shape_class), value)
    return out


def _declared_slots_floor(role: str) -> dict[int, int]:
    """Port -> `-np` for EVERY declared instance, independent of compile mode.

    `stack_manifest.declared_slots_by_port` reads NUMA_CONFIG, which lists a
    role's instances regardless of which NUMA mode is active, so it is the one
    source here that does not depend on how the compiler was invoked.

    Lazy import, mirroring `_launch_runtime_record` below: importing
    scripts.server.stack_manifest at module scope re-enters this module and
    yields "partially initialized module". Failure degrades to {} — the caller
    still produces the mode-scoped map, which is what it produced before.
    """
    try:
        from scripts.server.stack_manifest import declared_slots_by_port

        declared = declared_slots_by_port(role)
        if not declared:
            # ALIAS roles have no NUMA_CONFIG entry of their own — they ride a
            # host's process — so their floor is empty and they keep only the
            # single port their launch entries tagged. That left server_8082 /
            # server_8182 still attesting against a role-level 16 after the
            # host's own map was fixed. An alias serves its host's whole fleet,
            # so it inherits the host's declared map.
            aliases, _roles = _stack_manifest_info()
            host = (aliases or {}).get(role)
            if isinstance(host, str) and host and host != role:
                declared = declared_slots_by_port(host)
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(declared, dict):
        return {}
    return {
        int(port): int(slots)
        for port, slots in declared.items()
        if isinstance(port, int)
        and isinstance(slots, int)
        and not isinstance(slots, bool)
        and slots > 0
    }


def _slots_by_port(
    role: str,
    launch: dict[str, Any],
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
    role_slots: int | None,
    port_shape_classes: dict[int, str] | None = None,
) -> dict[int, int]:
    """Port -> `-np`, one entry per launch instance. THE JOIN.

    `slots` is per-ROLE; the operator's spec is per-INSTANCE SHAPE. A role's full
    96-core instance and its 48-core halves want different slot counts, and one
    number cannot say that. Each launch entry already carries the SHAPE CLASS its
    `cpu_shape` resolves to (placement, from stack_topology.yaml); this maps that
    class through the registry's `slots_by_shape` (model, from server_mode) to a
    per-port answer. A class the role does not declare falls back to the role's
    single slot count, which is how the two single-instance GPU roles work.
    """
    by_shape = _slots_by_shape_declaration(server_cfg, role_cfg)
    out: dict[int, int] = {}

    # Port -> shape class over the role's WHOLE serving fleet. For an alias this
    # is the host's fleet, which is wider than the alias's own tagged entries —
    # the entries alone would leave the untagged instances on the role-level
    # fallback and claim a full's slot count on a half. Entries are merged in
    # afterwards so a launcher-only lane with no fleet map still resolves.
    classes: dict[int, str] = {}
    if isinstance(port_shape_classes, dict):
        for port, shape_class in port_shape_classes.items():
            if isinstance(port, int) and isinstance(shape_class, str) and shape_class:
                classes[port] = shape_class
    entries = launch.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            port = entry.get("port")
            shape_class = entry.get("cpu_shape_class")
            if isinstance(port, int):
                if isinstance(shape_class, str) and shape_class:
                    classes.setdefault(port, shape_class)
                else:
                    classes.setdefault(port, "")

    for port, shape_class in classes.items():
        value = by_shape.get(shape_class) if shape_class else None
        if value is None:
            value = role_slots
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            out[port] = value

    # FLOOR: every DECLARED instance, not just those the current compile mode
    # emitted. `classes` above comes from the launch view, which is filtered by
    # the NUMA mode the compiler ran under — so priors compiled in `full` mode
    # carried only {8070: 16} while the fleet ran `both` and bound 8080/8180 at
    # -np 4. Runtime attestation looks up the LIVE `--port`, missed, fell back to
    # the role-level `slots` of 16, and reported drift on four correctly-launched
    # half instances. That is exactly the outcome this function's docstring says
    # it exists to prevent, and a warning channel that cries wolf on correct
    # servers is worse than none.
    #
    # A port -> slots LOOKUP is keyed, so carrying ports that this mode does not
    # launch costs nothing and makes attestation right in every mode. Mode-scoped
    # values WIN on conflict: they reflect what the compiler actually resolved
    # for a port it emitted; the floor only fills ports it never considered.
    for port, slots in _declared_slots_floor(role).items():
        out.setdefault(port, slots)

    return {port: out[port] for port in sorted(out)}


def _launch_runtime_record(
    role: str,
    descriptor: dict[str, Any],
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
    launch_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(launch_cfg, dict):
        return {}
    launch = launch_cfg.get("launch")
    if not isinstance(launch, dict):
        return {}

    try:
        from scripts.server.stack_manifest import (
            DEFAULT_EFFECTIVE_CONTEXT_TOKENS,
            DEFAULT_UBATCH_TOKENS,
            LAUNCH_KV_QUANT_CONFIGS,
            NO_SPEC_DECODE_ROLES,
            # SERIAL_ROLES is deliberately NOT imported any more — see the
            # "THE SERIAL_ROLES CLAMP IS GONE" note below. Serialisation is an
            # admission input; this module compiles SERVING numbers.
            VISION_ESCALATION_DEVICE,
            VISION_ESCALATION_REASONING,
            fallback_slots_for_mode,
        )
        from scripts.server.stack_numa import MLOCK_ROLES
        from scripts.server.stack_paths import (
            LLAMA_SERVER,
            LLAMA_SERVER_V2,
            SLOT_SAVE_DIR,
            _PATHS,
            _V2_ROLES,
        )
        from src.roles import Role
    except Exception:
        return {}

    descriptor_serving = descriptor.get("serving") if isinstance(descriptor.get("serving"), dict) else {}
    primary_role = _first_string(launch.get("primary_roles")) or role
    mode = _first_string(launch.get("modes")) or "default"
    worker_type = _first_launch_entry_value(launch, "worker_type")
    vision_type = _first_launch_entry_value(launch, "vision_type")
    requirements = launch.get("requirements") if isinstance(launch.get("requirements"), dict) else {}
    acceleration = _effective_acceleration(role_cfg, server_cfg)
    binary_dir, ld_paths = _runtime_requirements(server_cfg, role_cfg)

    # No explicit binary_dir? Derive one from the role's declared DEVICE via the
    # stable kernel layer, instead of falling through to a CPU-only literal. This
    # is what makes `device: ROCm0` reach the launcher: without it a GPU role
    # silently ran on llama.cpp/build/bin, which has no libggml-hip.so.
    backend = _backend_for_role(server_cfg, role_cfg)
    # An EXPLICIT registry binary_dir is an override and carries env consequences
    # (strip-ggml policy, KMP_BLOCKTIME). A backend-derived default must not, or
    # every role silently changes env policy. Keep the two distinguishable.
    explicit_binary_override = bool(binary_dir)
    if not binary_dir:
        # FAIL CLOSED. This used to swallow every exception and leave binary_dir
        # None, which fell through to the CPU-only `LLAMA_SERVER` literal below —
        # so a dangling or mis-pointed `production/<backend>` symlink after a
        # promotion compiled a GPU role onto a CPU build, silently, with nothing
        # downstream able to tell. The ONLY legitimate "no store consulted" case is
        # an EXPLICIT registry `runtime_requirements.binary_dir`, and that is the
        # `if not binary_dir` guard above — not a catch-all here.
        from src.registry.kernel_paths import (
            KernelPathError as _KernelPathError,
            backend_dir as _kernel_backend_dir,
        )

        try:
            binary_dir = str(_kernel_backend_dir(backend))
        except _KernelPathError as exc:
            raise _KernelPathError(
                f"role {role!r} resolves to kernel backend {backend!r}, which does not "
                f"resolve: {exc}. Declare runtime_requirements.binary_dir for this role "
                f"or repoint the production kernel store."
            ) from exc

    # 2026-08-01 (INC vision `invalid device: ROCm0`): resolving binary_dir from the
    # backend was only HALF the derivation. The binary is chosen by backend but its
    # ggml was still chosen by whatever LD_LIBRARY_PATH the launching shell happened
    # to export — on this host that is the CPU tree, which outranks the binary's own
    # RUNPATH. The HIP binary therefore loaded CPU-only ggml, registered no ROCm
    # device, and rejected its own declared `--device ROCm0`. Derive the library path
    # from the SAME backend that chose the binary, so the two can never disagree.
    # An EXPLICIT registry ld_library_path stays authoritative (it is an override,
    # and the only reason to write one is to name something non-derivable).
    #
    # FAIL CLOSED here too: `ld_paths = []` on failure is not a neutral default, it
    # is the exact condition the 2026-08-01 incident describes — the HIP binary with
    # the CPU tree's ggml ahead of it on LD_LIBRARY_PATH. Note `[]` remains the
    # CORRECT resolved answer for the `cpu` backend (the ambient environment already
    # is that tree); what is gone is `[]` as the answer to "we could not tell".
    if not ld_paths:
        from src.registry.kernel_paths import (
            KernelPathError as _KernelPathError,
            backend_ld_library_path as _kernel_backend_ld,
        )

        try:
            ld_paths = _kernel_backend_ld(backend)
        except _KernelPathError as exc:
            raise _KernelPathError(
                f"role {role!r} resolves to kernel backend {backend!r}, whose "
                f"LD_LIBRARY_PATH does not resolve: {exc}. Declare "
                f"runtime_requirements.ld_library_path for this role or repoint the "
                f"production kernel store."
            ) from exc

    binary_path = str(Path(binary_dir) / "llama-server") if binary_dir else str(LLAMA_SERVER)
    binary_family = (
        str(descriptor_serving.get("binary"))
        if isinstance(descriptor_serving.get("binary"), str)
        else "llama.cpp-v2"
        if primary_role in _V2_ROLES and LLAMA_SERVER_V2.exists()
        else "llama.cpp"
    )
    if not binary_dir and primary_role in _V2_ROLES and LLAMA_SERVER_V2.exists():
        binary_path = str(LLAMA_SERVER_V2)

    lane_shape: dict[str, int] | None = None
    if mode == "gpu_shadow_lane":
        # gpu-serving-tie-in P2-6 (P0-1c): lane serving shape is DATA from the
        # np_ceiling policy's serving_shape block — the CPU-mode defaults
        # (2 slots, 32768-token context) are wrong for the lane. Missing/invalid
        # shape -> slots None (refused downstream via the launcher-tenant
        # known_gap), never a guessed CPU default.
        lane_shape = _gpu_shadow_lane_serving_shape()
        slots = lane_shape.get("np_slots") if isinstance(lane_shape, dict) else None
    else:
        # 2026-08-02 phase 2: the per-mode slot LITERALS that used to live here
        # (worker_pool 4/1, vision 1/2, embedding 4, default 1-if-serial-else-2)
        # were a second copy of the launcher's, and admission carries a third.
        # They now come from launch_shape.fallback_slots in launch_manifest.yaml.
        #
        # `fallback_slots_for_mode`, NOT `resolve_slots`: this compiler is handed
        # a `registry_path` and must resolve declarations against THAT registry,
        # which `_runtime_flag_int_prior` below already does. A second, ambient
        # master read here would give one field two sources.
        slots = fallback_slots_for_mode(
            mode, worker_type=worker_type, vision_type=vision_type
        ).slots

    # 2026-08-01: `slots` is DECLARED DATA, and the mode literals above were
    # shadowing it. `server_mode.worker_vision.slots: 1` never reached the compiled
    # record because the `mode == "vision"` branch hardcoded 2 for any non-escalation
    # VL server, so the launcher emitted `-np 2` for a role that declares 1.
    #
    # `_runtime_flag_int_prior` still runs on top of the resolver because it can see
    # declarations the resolver cannot: it reads the ALREADY-BOUND server_cfg plus
    # the role's own `server`/`serving`/`launch` sub-mappings, which is where a role
    # with no server_mode row of its own can still carry one.
    #
    # ── 2026-08-02: THE SERIAL_ROLES CLAMP IS GONE ──────────────────────────────
    # It used to sit here and force `slots = 1` for any role in SERIAL_ROLES, so
    # this record shipped two disagreeing slot counts: `serving.slots` (master's
    # declaration, which admission turned into a limit) versus
    # `runtime.cache.slots` (clamped, which the launcher turned into `-np`). That
    # was an ADMISSION policy applied to a SERVING number, and the note here said
    # removing it was "a measurement question, not a refactor" — correctly, at the
    # time, because nobody had decided what the slot counts should be.
    #
    # They have now been decided. The operator ratified explicit per-instance slot
    # counts (frontdoor full 16 / half 4, worker full 16 / half 4, critic 4, ingest
    # 4, 27B 8, VL 4) with the per-slot context each implies, and a clamp to 1
    # would silently discard every one of them. SERIAL_ROLES remains what it always
    # should have been: an input to admission in src/api/admission.py, not to `-np`.
    # `src/registry/stack_priors.live_stack_serving_slot_limits` now keys admission
    # PER PORT off these same per-instance values, which is what finally makes
    # admission.py's stated intent — "Aligned admission limits with llama-server
    # slot counts (no idle slots)" — true per instance rather than per role.
    #
    # gpu_shadow_lane is exempt: its shape is policy data from
    # gpu_shadow_lane_np_ceiling.yaml and a None there means "refuse", not "guess".
    if mode != "gpu_shadow_lane":
        declared_slots = _runtime_flag_int_prior(server_cfg, role_cfg, key="slots", minimum=1)
        if declared_slots is not None:
            slots = declared_slots

    canonical_primary_role = str(Role.from_string(primary_role, default=None) or primary_role)
    canonical_role = str(Role.from_string(role, default=None) or role)
    kv_types = _kv_types_prior(server_cfg, role_cfg, descriptor) or (
        LAUNCH_KV_QUANT_CONFIGS.get(canonical_primary_role)
        or LAUNCH_KV_QUANT_CONFIGS.get(canonical_role)
    )
    override_kv = _override_kv_args(acceleration)
    override_kv = sorted(set(override_kv))

    spec: dict[str, Any] = {
        "enabled": False,
        "type": None,
        "disabled_by": None,
        "draft_model_path": None,
        "draft_max": None,
        "draft_min": None,
        "draft_p_min": None,
        "draft_p_split": None,
        "threads_draft": None,
        "ngram_mod_n_min": None,
        "ngram_mod_n_max": None,
        "ngram_mod_n_match": None,
        "n_gpu_layers_draft": None,
    }
    # 2026-06-26 v6 cutover: spec_type carries the v6 MTP token 'draft-mtp' (bare
    # 'mtp' is invalid in v6). It is preserved verbatim from the registry
    # acceleration block — no normalization/allow-list rejects it here.
    #
    # 2026-07-31: spec_type may now be a COMPOSED, comma-separated chain such as
    # `ngram-mod,draft-mtp` (the production recipe — see the canonical
    # `speculative_decoding_policy` block in the master registry). The branch
    # below previously tested `== "draft-mtp"` exactly, so a composed value fell
    # through to the DISABLED spec and silently launched the role with NO
    # speculation at all — strictly worse than draft-mtp alone. Match on
    # membership of the chain instead of string equality.
    spec_type_prior = (
        str(acceleration.get("spec_type"))
        if isinstance(acceleration.get("spec_type"), str) and acceleration.get("spec_type")
        else None
    )
    if mode == "worker_pool" and worker_type == "explore":
        worker_draft_max = _positive_int_prior(
            acceleration,
            key="draft_max",
            fallback=2,
        )
        worker_threads_draft = _positive_int_prior(
            acceleration,
            key="threads_draft",
            fallback=16,
        )
        worker_draft_min = _nonnegative_int_prior(
            acceleration,
            key="draft_min",
        )
        worker_draft_p_min = _number_prior(
            acceleration,
            key="draft_p_min",
            fallback=0.0,
        )
        worker_draft_p_split = (
            _number_prior(acceleration, key="draft_p_split", fallback=0.0)
            if "draft_p_split" in acceleration
            else None
        )
        spec.update(
            {
                "enabled": True,
                # 2026-06-26 v6 cutover: prefer the registry spec_type (draft-mtp);
                # the literal fallback is degraded mode for incomplete registries.
                "type": spec_type_prior or "draft-mtp",
                "draft_model_path": str(requirements.get("draft_model_path"))
                if requirements.get("draft_model_path")
                else None,
                "draft_max": worker_draft_max,
                "draft_min": worker_draft_min,
                "draft_p_min": worker_draft_p_min,
                "draft_p_split": worker_draft_p_split,
                "threads_draft": worker_threads_draft,
                "ngram_mod_n_min": _nonnegative_int_prior(
                    acceleration,
                    key="ngram_mod_n_min",
                ),
                "ngram_mod_n_max": _nonnegative_int_prior(
                    acceleration,
                    key="ngram_mod_n_max",
                ),
                "ngram_mod_n_match": _positive_int_prior(
                    acceleration,
                    key="ngram_mod_n_match",
                    fallback=0,
                )
                if "ngram_mod_n_match" in acceleration
                else None,
            }
        )
    elif _spec_type_launches_drafter(spec_type_prior) and role == primary_role:
        # 2026-06-26 v6 cutover: emit a NON-NULL draft-mtp spec ONLY for the PRIMARY
        # role that launches the server (role == primary_role). ALIAS roles
        # (shared_with_first_n, e.g. coder_escalation / worker_summarize sharing
        # frontdoor's :8070 process) inherit the host's NEXTN draft at runtime and do
        # NOT launch their own draft — they fall through to the disabled spec so their
        # launch record matches the launch manifest (which nulls draft for aliases).
        # emit a NON-NULL draft-mtp spec for any non-worker
        # role whose registry acceleration.spec_type == 'draft-mtp' (frontdoor
        # qwen36_q8_0, architect_general). These are NEXTN self-draft models — the
        # draft head is embedded in the base GGUF, so the resolved drafter file is
        # the same file as -m. Keep the compiled path explicit for provenance; the
        # launcher suppresses -md for same-realpath drafts and preserves
        # --spec-type/--spec-draft-n-max. draft_max carries the n-max value from
        # the registry (frontdoor=4, architect=4).
        nextn_draft_path = _resolve_nextn_draft_path(
            requirements,
            acceleration,
            server_cfg,
            models_dir=_PATHS.get("models_dir"),
        )
        if not _spec_type_has_mtp(spec_type_prior):
            # An EXTERNAL drafter must be a DIFFERENT file. The resolver falls back to the
            # model path (NEXTN self-draft) and the launcher suppresses -md for a
            # same-realpath draft, so a draft-dflash role with no resolvable drafter would
            # launch `--spec-type draft-dflash` with no -md and die at load. Refuse here,
            # at compile, instead of at HIP load time in production.
            model_for_role = requirements.get("model_path")
            if not nextn_draft_path or (
                isinstance(model_for_role, str)
                and os.path.realpath(str(nextn_draft_path)) == os.path.realpath(model_for_role)
            ):
                raise ValueError(
                    f"role {role!r}: spec_type {spec_type_prior!r} needs an external drafter "
                    f"GGUF distinct from the model; resolved draft {nextn_draft_path!r}. "
                    "Declare it in the master roles.<model_role>.drafters recipe."
                )
        draft_max_prior = acceleration.get("draft_max")
        # `-ngld` for an EXTERNAL drafter (DFlash2 recipe: ngld 99). Declared only by
        # the projected drafter recipe; absent -> not emitted (llama-server default).
        ngld_prior = acceleration.get("n_gpu_layers_draft")
        spec.update(
            {
                "enabled": True,
                "type": spec_type_prior,
                "n_gpu_layers_draft": ngld_prior
                if (isinstance(ngld_prior, int) and not isinstance(ngld_prior, bool))
                or (isinstance(ngld_prior, str) and ngld_prior in ("all", "auto"))
                else None,
                "draft_model_path": str(nextn_draft_path) if nextn_draft_path else None,
                "draft_max": draft_max_prior
                if isinstance(draft_max_prior, int) and not isinstance(draft_max_prior, bool)
                else None,
                "draft_min": _nonnegative_int_prior(
                    acceleration,
                    key="draft_min",
                ),
                "draft_p_min": (
                    _number_prior(acceleration, key="draft_p_min", fallback=0.0)
                    if "draft_p_min" in acceleration
                    else None
                ),
                "draft_p_split": (
                    _number_prior(acceleration, key="draft_p_split", fallback=0.0)
                    if "draft_p_split" in acceleration
                    else None
                ),
                "ngram_mod_n_min": _nonnegative_int_prior(
                    acceleration,
                    key="ngram_mod_n_min",
                ),
                "ngram_mod_n_max": _nonnegative_int_prior(
                    acceleration,
                    key="ngram_mod_n_max",
                ),
                "ngram_mod_n_match": _positive_int_prior(
                    acceleration,
                    key="ngram_mod_n_match",
                    fallback=0,
                )
                if "ngram_mod_n_match" in acceleration
                else None,
            }
        )
    elif primary_role in NO_SPEC_DECODE_ROLES and (
        acceleration.get("draft_role")
        or acceleration.get("draft_max")
        or acceleration.get("n_layer_exit_draft")
    ):
        spec["disabled_by"] = "no_spec_decode"

    # `-c`. 2026-08-02: the HANDED registry's `serving_shape.n_ctx` wins, exactly
    # as `slots` does. `launch_cfg["effective_context_tokens"]` is the launcher's
    # view (stack_manifest derives it from the AMBIENT registry) and stays as the
    # fallback for roles the handed registry says nothing about — every alias, and
    # the launcher-only lanes.
    declared_n_ctx = _runtime_flag_int_prior(server_cfg, role_cfg, key="n_ctx", minimum=1)
    context_tokens = declared_n_ctx
    if not isinstance(context_tokens, int):
        context_tokens = launch_cfg.get("effective_context_tokens")
    if not isinstance(context_tokens, int):
        context_tokens = DEFAULT_EFFECTIVE_CONTEXT_TOKENS
    if mode == "worker_pool" and worker_type == "explore" and declared_n_ctx is None:
        # `roles.<role>.model.max_context` is a MODEL-CAPABILITY fallback, and it
        # OVERRIDES rather than caps — so while it was unconditional it silently
        # beat an explicit serving declaration. worker_general is the live case:
        # `max_context: 16384` is a stale deployment shape (the gemma4 GGUF's own
        # `gemma4.context_length` is 262144), and it would have held -c at 16384
        # while `serving_shape.n_ctx: 262144` was ignored, giving 1024 tokens per
        # slot at -np 16. An explicit declaration now wins; the capability figure
        # still applies to any worker role that has no serving_shape.
        context_tokens = _worker_context_prior(role_cfg, fallback=context_tokens)
    elif mode == "gpu_shadow_lane" and isinstance(lane_shape, dict):
        # P0-1c: total -c = np_slots * slot_context_tokens from the serving_shape
        # block (policy data), not the launcher's CPU-mode context default.
        shape_context = lane_shape.get("context_tokens")
        if isinstance(shape_context, int) and shape_context > 0:
            context_tokens = shape_context
    worker_ubatch = _positive_int_prior(
        acceleration,
        key="ubatch",
        fallback=512,
    )
    # `--cache-idle-slots` / `--no-cache-idle-slots` (llama-server, default on when
    # --cache-ram > 0). Declared as `server_mode.<role>.cache_idle_slots`, beside
    # cache_ram; compiled ONLY when declared (None -> the launcher emits nothing), so
    # every other role's compiled record is byte-identical. STACKCHG-8083BATCH-20261004.
    cache_idle_slots = _runtime_flag_bool_prior(server_cfg, role_cfg, key="cache_idle_slots")
    # A default-mode server may DECLARE its -ub/-b (serving_shape.ubatch / .batch).
    # Undeclared -> DEFAULT_UBATCH_TOKENS and no -b, exactly as before; the declared
    # batch is compiled ONLY when declared. STACKCHG-8083BATCH-20261004 part B.
    default_ubatch: int | None = None
    default_batch: int | None = None
    if mode == "default":
        default_ubatch, default_batch = _declared_batch_shape_prior(
            role, server_cfg, role_cfg, default_ubatch=DEFAULT_UBATCH_TOKENS
        )

    return {
        "binary_family": binary_family,
        "binary_path": binary_path,
        "binary_dir": binary_dir,
        "ld_library_path": ld_paths,
        "env_policy": (
            "binary_override_strip_ggml" if explicit_binary_override else "canonical"
        ),
        "kmp_blocktime": 10 if explicit_binary_override else None,
        "cache": {
            "context_tokens": context_tokens,
            # Role-level default. Kept because plenty of consumers only want "how
            # many slots does this role run"; the per-instance answer is below.
            "slots": slots,
            # Port -> `-np` for THIS role's instances. The launcher indexes it by
            # the port it is about to bind, and admission keys its per-endpoint
            # limits off the same map, so `admission_limit == -np` holds per
            # instance rather than per role. Empty for a role with no launch
            # entries (nothing to place).
            "slots_by_port": _slots_by_port(
                role,
                launch,
                server_cfg,
                role_cfg,
                slots,
                launch_cfg.get("port_shape_classes")
                if isinstance(launch_cfg, dict)
                else None,
            ),
            "ubatch": worker_ubatch
            if mode == "worker_pool" and worker_type == "explore"
            else default_ubatch
            if mode == "default"
            else None,
            # Logical batch (-b). Present only when declared (see above).
            **({"batch": default_batch} if default_batch is not None else {}),
            "kv_type_k": kv_types[0] if kv_types else None,
            "kv_type_v": kv_types[1] if kv_types else None,
            "kv_hadamard": bool(primary_role in _V2_ROLES and LLAMA_SERVER_V2.exists()),
            # Unified KV pool (llama-server -kvu / --no-kv-unified). None when
            # undeclared -> the launcher emits nothing and the kernel default
            # applies (split KV whenever -np is explicit, which it always is).
            # Declared as `server_mode.<role>.serving_shape.kv_unified`: with it,
            # ONE request may use the whole -c pool; without it each slot is
            # hard-capped at -c / -np. See _runtime_flag_bool_prior.
            "kv_unified": _runtime_flag_bool_prior(server_cfg, role_cfg, key="kv_unified")
            if mode == "default"
            else None,
            # Speculative DRAFT-context KV types (llama-server -ctkd/-ctvd). -ctk/-ctv
            # do NOT reach the draft context: common_base_params_to_speculative copies
            # speculative.draft.cache_type_{k,v}, which default to F16. Declared as
            # `serving_shape.draft_kv_quant: {k, v}`; None when undeclared.
            "draft_kv_type_k": (_draft_kv_types_prior(server_cfg, role_cfg) or (None, None))[0],
            "draft_kv_type_v": (_draft_kv_types_prior(server_cfg, role_cfg) or (None, None))[1],
            # 2026-06-26 v6 cutover: no_mmap is no longer hardcoded to worker_pool+explore.
            # It now flows through from the role's config (server_mode then roles block),
            # defaulting to False when absent — so non-worker quarter roles (N12 private
            # quarters) can request no_mmap=True without forcing it globally. The legacy
            # worker_pool+explore canonical-recipe default is preserved as a fallback.
            "no_mmap": _role_no_mmap_prior(
                server_cfg,
                role_cfg,
                default=bool(mode == "worker_pool" and worker_type == "explore"),
            ),
            "mlock": bool(mode == "default" and primary_role in MLOCK_ROLES),
            "slot_save_path": str(SLOT_SAVE_DIR / primary_role) if mode == "default" else None,
        },
        "flags": {
            "flash_attn": True,
            # 2026-06-26: architect_general no longer excluded from --jinja. The
            # 2026-04-15 exclusion (commit 0879ed56) suppressed Qwen3.5-122B hybrid
            # <think>-loops by falling back to generic ChatML, but that also made the
            # registry's enable_thinking=false inert (kwarg only applies on the
            # /v1/chat/completions+jinja path). Enrolling architect into jinja routes
            # it through chat-completions where nothink fires (frontdoor proves the
            # same-family draft-mtp+jinja+nothink path). Gated on the J12 think-loop
            # suppression probe before trusting.
            "jinja": bool(
                (mode == "default")
                or (mode == "worker_pool" and worker_type == "explore")
            ),
            "device": _runtime_flag_string_prior(
                server_cfg,
                role_cfg,
                key="device",
                fallback=VISION_ESCALATION_DEVICE
                if mode == "vision" and vision_type == "escalation"
                else None,
            ),
            # Separate draft device, when a role declares one. None everywhere today;
            # the launcher then lets the draft follow the target's device rather than
            # stranding a NEXTN self-draft on the CPU under a GPU-resident target.
            "device_draft": _runtime_flag_string_prior(
                server_cfg, role_cfg, key="device_draft"
            ),
            # 2026-08-01: `device` alone does not put a model on the GPU. The three
            # flags below were declared in the registry's serving block and stopped
            # here — never compiled, never emitted — so the VL server was asked to
            # use ROCm0 with zero offloaded layers, full-resolution image tokens and
            # the prompt cache on. All three are None when undeclared: they are
            # GPU-shaped, and a launcher default would apply them to CPU roles too.
            # `cache_ram` in particular is meaningfully 0 ("disable the prompt
            # cache"), so absence must be None and not 0.
            "n_gpu_layers": _n_gpu_layers_prior(server_cfg, role_cfg),
            "image_min_tokens": _runtime_flag_int_prior(
                server_cfg, role_cfg, key="image_min_tokens"
            ),
            "cache_ram": _runtime_flag_int_prior(server_cfg, role_cfg, key="cache_ram"),
            # Companion of cache_ram: whether idle slots are saved to the prompt cache
            # (and, on a unified pool, cleared from it) when a new task starts. Present
            # only when declared. STACKCHG-8083BATCH-20261004.
            **({"cache_idle_slots": cache_idle_slots} if cache_idle_slots is not None else {}),
            # Per-role chat template FILE (llama-server --chat-template-file).
            # Declared as `server_mode.<role>.chat_template_file` (absolute path
            # string); role-local server/serving/launch sub-mappings work too via
            # the shared string-prior search order. None when undeclared — the
            # launcher then emits nothing and the GGUF-embedded template applies.
            # No default here: a template file is model-specific, and a compiler
            # default would silently re-template every role.
            "chat_template_file": _runtime_flag_string_prior(
                server_cfg, role_cfg, key="chat_template_file"
            ),
            "reasoning": _runtime_reasoning_prior(
                server_cfg,
                role_cfg,
                descriptor,
                mode=mode,
                worker_type=worker_type,
                vision_type=vision_type,
                vision_escalation_reasoning=VISION_ESCALATION_REASONING,
            ),
            "override_kv": override_kv,
            "spec": spec,
        },
    }


def _models_dir() -> Path:
    try:
        from scripts.server.stack_paths import _PATHS

        models_dir = _PATHS.get("models_dir")
        if models_dir:
            return Path(models_dir)
    except Exception:
        pass
    return DEFAULT_MODELS_DIR


def _model_base_dir() -> Path:
    try:
        from src.registry.registry_loader import RegistryLoader

        return RegistryLoader(validate_paths=False).model_base_path
    except Exception:
        return DEFAULT_MODEL_BASE_DIR


def _resolved_model_path(value: Any, *, base_dir: Path | None = None) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir or _models_dir()) / path)


def _role_model_path(role_cfg: dict[str, Any] | None) -> str | None:
    model_cfg = _nested_mapping(role_cfg, "model")
    if not isinstance(model_cfg, dict):
        return None
    return _resolved_model_path(model_cfg.get("path"), base_dir=_model_base_dir())


def _server_mode_launch_requirement_overrides(
    role: str,
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
) -> dict[str, str]:
    if not isinstance(server_cfg, dict):
        return {}

    overrides: dict[str, str] = {}
    explicit_model = _resolved_model_path(server_cfg.get("model_path"))
    server_model_role = server_cfg.get("model_role")
    role_model = (
        _role_model_path(role_cfg)
        if not isinstance(server_model_role, str) or server_model_role == role
        else None
    )
    server_model = _resolved_model_path(server_cfg.get("model"))
    if explicit_model or role_model or server_model:
        overrides["model_path"] = str(explicit_model or role_model or server_model)

    explicit_draft = _resolved_model_path(server_cfg.get("draft_model_path"))
    server_draft = _resolved_model_path(server_cfg.get("draft_model"))
    if explicit_draft or server_draft:
        overrides["draft_model_path"] = str(explicit_draft or server_draft)

    mmproj_path = _resolved_model_path(server_cfg.get("mmproj_path"))
    if mmproj_path:
        overrides["mmproj_path"] = mmproj_path
    return overrides


def _role_memory_cost(
    role: str,
    role_cfg: dict[str, Any] | None,
    server_cfg: dict[str, Any] | None,
) -> tuple[float | None, str | None, list[str]]:
    gaps: list[str] = []
    if isinstance(server_cfg, dict):
        cost = _residency_cost(server_cfg.get("tier"))
        if cost is not None:
            return cost, "server_mode.tier", gaps

    if isinstance(role_cfg, dict):
        memory = role_cfg.get("memory")
        if isinstance(memory, dict):
            cost = _residency_cost(memory.get("residency"))
            if cost is not None:
                return cost, "roles.memory.residency", gaps

    gaps.append("Missing memory residency evidence")
    return None, None, gaps


def _serving_record(
    role: str,
    descriptor: dict[str, Any],
    role_cfg: dict[str, Any] | None,
    server_role: str | None,
    server_cfg: dict[str, Any] | None,
    binding: str,
    launch_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    serving = descriptor.get("serving")
    descriptor_serving = serving if isinstance(serving, dict) else {}
    ports: set[int] = set()
    launch_ports = (
        [port for port in launch_cfg.get("ports", []) if isinstance(port, int)]
        if isinstance(launch_cfg, dict)
        else []
    )
    if launch_ports:
        ports.update(launch_ports)
    else:
        for value in descriptor_serving.get("ports") or []:
            if isinstance(value, int):
                ports.add(value)
    if isinstance(server_cfg, dict):
        slots = server_cfg.get("slots")
        if not launch_ports:
            port = server_cfg.get("port")
            if isinstance(port, int):
                ports.add(port)
            numa_ports = server_cfg.get("numa_ports")
            if isinstance(numa_ports, list):
                ports.update(port for port in numa_ports if isinstance(port, int))
    else:
        slots = None

    launch_record = (
        copy.deepcopy(launch_cfg.get("launch") or _launch_record([]))
        if isinstance(launch_cfg, dict)
        else _launch_record([])
    )
    requirement_overrides = (
        _server_mode_launch_requirement_overrides(role, server_cfg, role_cfg)
        if isinstance(launch_cfg, dict)
        else {}
    )
    if requirement_overrides:
        requirements = launch_record.get("requirements")
        if not isinstance(requirements, dict):
            requirements = {}
        launch_record["requirements"] = {**requirements, **requirement_overrides}
    runtime_launch_cfg = copy.deepcopy(launch_cfg) if isinstance(launch_cfg, dict) else None
    if isinstance(runtime_launch_cfg, dict):
        runtime_launch_cfg["launch"] = launch_record
    runtime_record = _launch_runtime_record(
        role,
        descriptor,
        server_cfg,
        role_cfg,
        runtime_launch_cfg,
    )
    launch_record["runtime"] = runtime_record

    # Per-instance `-np`, stamped onto the entries themselves so a launch entry is
    # self-describing: port, placement (numa_instance + cpu_shape_class) and the
    # slot count that placement resolved to, in one row. `runtime.cache.slots` is
    # still the role-level default; these are what actually get launched.
    runtime_cache = runtime_record.get("cache") if isinstance(runtime_record, dict) else None
    slots_by_port = (
        runtime_cache.get("slots_by_port") if isinstance(runtime_cache, dict) else None
    )
    if isinstance(slots_by_port, dict):
        for entry in launch_record.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            entry_slots = slots_by_port.get(entry.get("port"))
            if isinstance(entry_slots, int) and entry_slots > 0:
                entry["slots"] = entry_slots

    if not isinstance(slots, int) or slots <= 0:
        cache = runtime_record.get("cache") if isinstance(runtime_record, dict) else {}
        runtime_slots = cache.get("slots") if isinstance(cache, dict) else None
        if isinstance(runtime_slots, int) and runtime_slots > 0:
            slots = runtime_slots

    sorted_ports = sorted(ports)
    if sorted_ports:
        endpoint = f"http://localhost:{sorted_ports[0]}"
    elif isinstance(launch_cfg, dict) and isinstance(launch_cfg.get("url"), str):
        endpoint = launch_cfg.get("url")
    elif isinstance(server_cfg, dict):
        endpoint = server_cfg.get("url")
    else:
        endpoint = None

    return {
        "endpoint": endpoint,
        "server_role": server_role,
        "binding": binding,
        "ports": sorted_ports,
        "slots": slots if isinstance(slots, int) and slots > 0 else None,
        "tier": launch_cfg.get("tier")
        if isinstance(launch_cfg, dict) and launch_cfg.get("tier") is not None
        else server_cfg.get("tier")
        if isinstance(server_cfg, dict)
        else None,
        "effective_context_tokens": launch_cfg.get("effective_context_tokens")
        if isinstance(launch_cfg, dict)
        and isinstance(launch_cfg.get("effective_context_tokens"), int)
        else None,
        "binary": descriptor_serving.get("binary"),
        "binary_dir": descriptor_serving.get("binary_dir"),
        "numa_policy": descriptor_serving.get("numa_policy"),
        "shared_mmap": bool(
            (descriptor.get("role_bindings") or {}).get("shared_mmap")
        )
        if isinstance(descriptor.get("role_bindings"), dict)
        else False,
        "launch": launch_record,
    }


def _policy_hints(
    serving_record: dict[str, Any],
    model_record: dict[str, Any],
) -> dict[str, Any]:
    """Project tap/high-cost/contention/lock policy hints for one role.

    Consumers (inference lock/tap, approval gate, contention scheduler) may read
    these generated classifications directly and keep any local table as an
    explicit degraded fallback/override only. Boolean memory-threshold hints are
    ``None`` when model memory evidence is missing rather than silently False.
    """
    shared_worker_launch = stack_prior_uses_shared_worker_launch(
        {"serving": serving_record}
    )
    mem_gb = model_record.get("mem_gb")
    mem_val = float(mem_gb) if isinstance(mem_gb, (int, float)) else None
    tap_safe_non_stream = (
        None if mem_val is None else mem_val >= POLICY_TAP_SAFE_NON_STREAM_MIN_MEM_GB
    )
    high_cost = None if mem_val is None else mem_val >= POLICY_HIGH_COST_MIN_MEM_GB
    return {
        # Inference-lock class: shared-worker launches take a shared lock; every
        # other role takes an exclusive lock (mirrors live_stack_lock_role_sets).
        "lock_class": "shared" if shared_worker_launch else "exclusive",
        # Cross-role contention class follows the same shared/exclusive split.
        "contention_class": "light" if shared_worker_launch else "heavy",
        # Tap safe-mode non-stream hint (True when the model is large enough that
        # streaming risks contention); None when model memory is unknown.
        "tap_safe_non_stream": tap_safe_non_stream,
        # Approval-gate high-cost hint (True for architect-tier memory footprint);
        # None when model memory is unknown.
        "high_cost": high_cost,
        "model_mem_gb": mem_val,
        "source": "stack_priors.compile",
        "thresholds": {
            "tap_safe_non_stream_min_mem_gb": POLICY_TAP_SAFE_NON_STREAM_MIN_MEM_GB,
            "high_cost_min_mem_gb": POLICY_HIGH_COST_MIN_MEM_GB,
        },
    }


def _role_record(
    role: str,
    descriptor: dict[str, Any],
    registry_roles: dict[str, Any],
    server_mode: dict[str, Any],
    stack_aliases: dict[str, str],
    stack_roles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    role_cfg = registry_roles.get(role)
    if not isinstance(role_cfg, dict):
        role_cfg = None
    server_role, server_cfg, binding = _server_for_role(
        role, server_mode, stack_aliases, stack_roles
    )
    launch_cfg = stack_roles.get(role)
    memory_cost, memory_source, memory_gaps = _role_memory_cost(role, role_cfg, server_cfg)
    known_gaps = [str(gap) for gap in descriptor.get("known_gaps") or []]
    gaps = list(dict.fromkeys(known_gaps + memory_gaps))

    throughput = _throughput_prior(descriptor, server_cfg)
    quality = _quality_prior(descriptor)
    if throughput is None:
        gaps.append("Missing throughput prior")
    quality_advisories: list[str] = []
    if quality is None:
        # A missing `overall` is only a GAP when there is no quality evidence at
        # all. A model measured on domain axes but never on the whole canonical
        # suite is honestly described by quality_overall=null, and demanding a
        # scalar here is what produced the defect this replaced: to satisfy this
        # very gap, the descriptor builder promoted single-domain scores into
        # `overall`, so Qwen3-Next-80B carried a 27-question LONG-CONTEXT score
        # and Qwen3-VL carried an MMMU VISION score as if each described the
        # whole model. The rule now asks "is there evidence?", not "is there a
        # scalar?" — and null-by-design is an ADVISORY, not a blocking gap,
        # because blocking is the pressure that manufactures fake numbers.
        evidence = _quality_evidence_axes(descriptor)
        if evidence:
            quality_advisories.append(
                "No whole-suite quality prior; domain evidence only "
                f"({', '.join(evidence)}). quality_overall is null BY DESIGN — "
                "read the per-axis evidence instead of expecting a scalar."
            )
        else:
            gaps.append(
                "Missing quality evidence entirely (no suite_vector, no public benchmarks)"
            )
    if server_cfg is None:
        gaps.append("Missing live server binding")
    architecture = descriptor.get("architecture")
    architecture = architecture if isinstance(architecture, dict) else {}
    model_record = {
        "family": descriptor.get("family"),
        "arch": descriptor.get("arch"),
        "params_b": descriptor.get("params_b"),
        "active_b": descriptor.get("active_b"),
        "quant": descriptor.get("quant"),
        "mem_gb": descriptor.get("mem_gb"),
        "ctx_max": descriptor.get("ctx_max"),
        "ctx_model_max": descriptor.get("ctx_model_max"),
        "modalities": copy.deepcopy(descriptor.get("modalities") or []),
    }
    for key in ("n_layers", "attention_layers"):
        if architecture.get(key) is not None:
            model_record[key] = architecture[key]

    serving_record = _serving_record(
        role,
        descriptor,
        role_cfg,
        server_role,
        server_cfg,
        binding,
        launch_cfg,
    )

    # gpu-serving-tie-in P2-6 (P0-1): a launcher-only entry's TENANT role
    # (tenant_role meta key) compiles with its own deployment_status so
    # live_stack consumers (routing, locks, attestation) never classify it
    # live; only an explicitly-requested launcher-only start resolves it.
    launcher_tenant = bool(
        isinstance(launch_cfg, dict) and launch_cfg.get("launcher_only_tenant")
    )
    if launcher_tenant:
        launch_runtime = stack_prior_launch({"serving": serving_record}).get("runtime")
        runtime_cache = (
            launch_runtime.get("cache") if isinstance(launch_runtime, dict) else None
        )
        if not isinstance(runtime_cache, dict) or not runtime_cache.get("slots"):
            gaps.append(
                "Missing launcher-tenant serving shape "
                "(np_ceiling policy serving_shape block unresolved)"
            )

    return {
        "role": role,
        "deployment_status": "launcher_tenant"
        if launcher_tenant
        else "live_stack"
        if role in stack_roles
        else "benchmark_or_candidate",
        "status": "compiled_with_gaps" if gaps else "compiled",
        "model_id": descriptor.get("model_id"),
        "display_name": descriptor.get("display_name"),
        "serving": serving_record,
        "policy": _policy_hints(serving_record, model_record),
        "priors": {
            "throughput_tps": throughput,
            # SCALAR, and null whenever no whole-suite measurement exists.
            # Never synthesised from a single domain — see quality_by_axis.
            "quality_overall": quality,
            # PER-AXIS evidence, benchmark-keyed. Two values are comparable IFF
            # they share a key; `long_context` under RULER-1M and `long_context`
            # under LongBench-v2 are different instruments and are kept under
            # their own benchmark names, never pooled.
            "quality_by_axis": _quality_axis_values(descriptor),
            # The figure a consumer scoring THIS ROLE should use: the
            # role-relevant axis when it can rank the fleet, otherwise the
            # universal aggregate, with `basis` stating which and why.
            "quality_for_role": _quality_for_role(
                descriptor, role_cfg, role, registry_roles, stack_aliases
            ),
            "quality_advisories": quality_advisories,
            "memory_cost": memory_cost,
        },
        "acceleration": copy.deepcopy(descriptor.get("acceleration") or {}),
        "model": model_record,
        "evidence": {
            "precedence": {
                "serving": "server_mode/stack_manifest outrank roles metadata",
                "memory_cost": memory_source,
                "spec": _portable_path(PRECEDENCE_SPEC),
            },
            "descriptor_server_roles": _descriptor_server_roles(descriptor),
            "alias_overrides": copy.deepcopy(
                (descriptor.get("role_bindings") or {}).get("alias_overrides") or []
            ),
            "quality": copy.deepcopy((descriptor.get("quality") or {}).get("measured", [])),
            "speed": copy.deepcopy((descriptor.get("speed") or {}).get("measured", [])),
        },
        "known_gaps": sorted(set(gaps)),
    }


def _default_roles_from_descriptors(descriptors: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    for descriptor in (descriptors.get("models") or []):
        if isinstance(descriptor, dict):
            roles.update(_descriptor_roles(descriptor))
    return roles


def compile_stack_priors(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    descriptor_path: Path = DEFAULT_DESCRIPTORS,
    active_roles: set[str] | None = None,
    allow_incomplete: bool = False,
    numa_mode: str | None = None,
    require_realized_mode: bool = False,
    connect: Callable[[str, int], bool] | None = None,
) -> dict[str, Any]:
    registry = _load_yaml(registry_path)
    descriptors = _load_yaml(descriptor_path)

    registry_roles = registry.get("roles") or {}
    server_mode = registry.get("server_mode") or {}
    if not isinstance(registry_roles, dict) or not isinstance(server_mode, dict):
        raise ValueError("registry must contain mapping-valued roles and server_mode sections")

    descriptor_by_role = _descriptor_by_role(descriptors)
    requested_roles = active_roles or _default_roles_from_descriptors(descriptors)
    # ESC-8 Fix 6: the WRITE/check compile path resolves the NUMA mode from the
    # realized fleet (or refuses) instead of reading the ambient default-full
    # env. Off by default so a plain compile keeps the legacy behavior.
    if numa_mode is None and require_realized_mode:
        numa_mode = _realized_compile_numa_mode(connect=connect)
    stack_aliases, stack_roles = _stack_manifest_info(numa_mode=numa_mode)
    role_records: dict[str, Any] = {}
    gaps_by_role: dict[str, list[str]] = {}

    for role in sorted(requested_roles):
        descriptor = descriptor_by_role.get(role)
        if descriptor is None:
            gaps_by_role[role] = ["Missing model descriptor binding"]
            continue
        record = _role_record(
            role,
            descriptor,
            registry_roles,
            server_mode,
            stack_aliases,
            stack_roles,
        )
        role_records[role] = record
        if record["known_gaps"]:
            gaps_by_role[role] = record["known_gaps"]

    if gaps_by_role and not allow_incomplete:
        raise StackPriorsCompileError(gaps_by_role)

    return {
        "stack_priors_version": STACK_PRIORS_VERSION,
        "contract": stack_priors_contract(),
        "compiled_at": _timestamp(),
        "status": "compiled_with_gaps" if gaps_by_role else "compiled",
        "coverage_scope": "descriptor_role_bindings"
        if active_roles is None
        else "explicit_active_roles",
        "precedence_spec": _portable_path(PRECEDENCE_SPEC),
        "source_artifacts": {
            "registry": _source_metadata(registry_path),
            "descriptors": _source_metadata(descriptor_path),
            "stack_manifest": _source_metadata(DEFAULT_STACK_MANIFEST),
            "stack_numa": _source_metadata(DEFAULT_STACK_NUMA),
            "launch_manifest": _source_metadata(DEFAULT_LAUNCH_MANIFEST),
            "stack_topology": _source_metadata(DEFAULT_STACK_TOPOLOGY),
            "orchestrator_stack": _source_metadata(DEFAULT_ORCHESTRATOR_STACK),
            "stack_paths": _source_metadata(DEFAULT_STACK_PATHS),
            "stack_runtime": _source_metadata(DEFAULT_STACK_RUNTIME),
        },
        "roles": role_records,
        "known_global_gaps": {
            role: list(gaps) for role, gaps in sorted(gaps_by_role.items()) if gaps
        },
    }


def write_stack_priors(
    output_path: Path,
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    descriptor_path: Path = DEFAULT_DESCRIPTORS,
    active_roles: set[str] | None = None,
    allow_incomplete: bool = False,
    numa_mode: str | None = None,
    require_realized_mode: bool = False,
    connect: Callable[[str, int], bool] | None = None,
) -> dict[str, Any]:
    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles=active_roles,
        allow_incomplete=allow_incomplete,
        numa_mode=numa_mode,
        require_realized_mode=require_realized_mode,
        connect=connect,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            priors,
            fh,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=200,
        )
    return priors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile derived stack priors")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--descriptors", type=Path, default=DEFAULT_DESCRIPTORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roles", nargs="+", help="Explicit role list")
    parser.add_argument("--dry-run", action="store_true", help="Print priors instead of writing")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Emit priors with known_gaps instead of refusing incomplete records",
    )
    args = parser.parse_args(argv)

    roles = set(args.roles) if args.roles else None
    priors = compile_stack_priors(
        registry_path=args.registry,
        descriptor_path=args.descriptors,
        active_roles=roles,
        allow_incomplete=args.allow_incomplete,
    )
    if args.dry_run:
        yaml.safe_dump(
            priors,
            sys.stdout,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=200,
        )
    else:
        write_stack_priors(
            args.output,
            registry_path=args.registry,
            descriptor_path=args.descriptors,
            active_roles=roles,
            allow_incomplete=args.allow_incomplete,
        )
        print(f"OK: wrote {len(priors.get('roles', {}))} role priors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
