"""Orchestrator stack manifest — ports, role aliases, model paths, classifications.

Extracted from orchestrator_stack.py during the 2026-05-22 Tranche-7 refactor.
Owns all the static configuration that describes WHAT the orchestrator should
launch and WHERE: PORT_MAP, HOT_ROLES, SERIAL_ROLES, ROLE_LAUNCH_META, model
paths, NUMA_REPLICA_PORTS, plus the classification helpers that compute
HOT_SERVERS / WARM_SERVERS from ROLE_LAUNCH_META + NUMA_CONFIG.

orchestrator_stack.py re-imports every name here, and the registry compiler's
`from orchestrator_stack import ROLE_LAUNCH_META` fallback path keeps working.

2026-08-01 — THIS MODULE IS NOW A THIN LOADER. Every table below used to be a
Python literal; they now live in `orchestration/launch_manifest.yaml` (launch
data) and `orchestration/stack_topology.yaml` (per-role NUMA wiring, via
stack_numa). The names, types and values are unchanged — `frozenset` is still
`frozenset`, tuples are still tuples — so no consumer changed.

The reason is not tidiness. The duplication this module accumulated (and the
ring of local fallback tables around it) existed BECAUSE it was code: a Python
table invites a second Python table beside it, and every severe launcher defect
found in the 2026-07 audit had the same shape — the compiled artifact was
correct and a local literal won anyway. Data does not invite that.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from scripts.server.stack_numa import (
    CPU_SHAPE_CLASSES,
    GPU_HOST_LANE,
    NUMA_CONFIG,
    NUMA_INSTANCE_SHAPE_CLASSES,
    _nodes_touched,
    _parse_cpus,
    instance_shape_class,
)
from scripts.server.stack_paths import LLAMA_MATH_TOOLS, _V2_ROLES, _PATHS


# =============================================================================
# Declared-data loader
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LAUNCH_MANIFEST_PATH = _REPO_ROOT / "orchestration" / "launch_manifest.yaml"
# ⚠ NAME IS A TRAP, KEPT ONLY FOR CALLER COMPATIBILITY. This is the orchestrator's
# COMPILED LEAN registry, NOT the research master at
# /mnt/raid0/llm/epyc-inference-research/orchestration/model_registry.yaml.
# The two have the same basename and the lean one is a projection of the other.
#
# Reading the compiled artifact is the CORRECT choice — it is what the orchestrator
# actually consumes, and master->lean is guarded separately by the compile chain.
# But the name and the docstring below both say "MASTER", and that cost a debugging
# cycle on 2026-08-02: an edit to the real master appeared to be ignored by the
# parity guard until `registry_compiler --force` had run. That is expected behaviour,
# not a bug — the guard simply could not see a change that had not been compiled yet.
#
# If you edit the research master, RECOMPILE before expecting this guard to react.
_LEAN_REGISTRY_PATH = _REPO_ROOT / "orchestration" / "model_registry.yaml"
_MASTER_REGISTRY_PATH = _LEAN_REGISTRY_PATH  # legacy alias; prefer _LEAN_REGISTRY_PATH

# Path placeholders the manifest may use. Substitution is literal, not
# str.format, so a value containing braces for some other reason is left alone
# instead of raising a KeyError at import.
_PATH_TOKENS: dict[str, str] = {
    "{models_dir}": str(_PATHS["models_dir"]),
    "{model_base}": str(_PATHS["model_base"]),
    "{project_root}": str(_PATHS["project_root"]),
    "{cache_dir}": str(_PATHS["cache_dir"]),
    "{llm_root}": str(_PATHS["llm_root"]),
    "{tmp_dir}": str(_PATHS["tmp_dir"]),
}

_REQUIRED_MANIFEST_SECTIONS = (
    "port_map",
    "optional_auxiliary_roles",
    "hot_roles",
    "serial_roles",
    "role_launch_meta",
    "models",
    "embedding",
    "vision",
    "launch_shape",
    "capacity",
    "gpu_shadow_lane",
    "orchestrator_profiles",
    "parity",
    "docker_services",
    "aux_services",
)


def _expand_paths(value):
    """Recursively expand {models_dir}-style placeholders in loaded strings."""
    if isinstance(value, str):
        for token, replacement in _PATH_TOKENS.items():
            if token in value:
                value = value.replace(token, replacement)
        return value
    if isinstance(value, list):
        return [_expand_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_paths(item) for key, item in value.items()}
    return value


def _load_launch_manifest(path: Path | None = None) -> dict:
    """Load orchestration/launch_manifest.yaml.

    No fallback table and no partial load. A launcher that cannot read its own
    manifest must fail at IMPORT rather than start something plausible — the
    failure mode this refactor exists to remove is precisely "a local default
    won over the declared value and the run looked fine".
    """
    manifest_path = _LAUNCH_MANIFEST_PATH if path is None else path
    try:
        document = yaml.safe_load(manifest_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"stack_manifest: declared launch manifest missing at {manifest_path}. "
            f"The launcher has no fallback manifest by design."
        ) from exc
    if not isinstance(document, dict):
        raise ValueError(f"stack_manifest: {manifest_path} did not parse to a mapping")
    missing = [key for key in _REQUIRED_MANIFEST_SECTIONS if key not in document]
    if missing:
        raise ValueError(f"stack_manifest: {manifest_path} is missing section(s) {missing}")
    return _expand_paths(document)


def _load_master_registry(path: Path | None = None) -> dict:
    """Load the COMPILED LEAN registry — master's projection, not master itself.

    See the note on ``_LEAN_REGISTRY_PATH``: same basename as the research master,
    different file. Edits to the research master are invisible here until
    ``registry_compiler --force`` has run.

    Same contract as ``_load_launch_manifest``: no fallback, no partial load.
    Phase 2 made the launcher READ master for `slots`, `device` and the role
    aliases instead of restating them, so master is now a hard input to the
    launcher's configuration rather than something it is merely cross-checked
    against. A launcher that cannot read it must fail at import; guessing is
    the failure mode the whole refactor exists to remove.
    """
    registry_path = _MASTER_REGISTRY_PATH if path is None else path
    try:
        document = yaml.safe_load(registry_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"stack_manifest: master registry missing at {registry_path}. The launcher "
            f"derives slots/device/aliases from it and has no fallback copy by design."
        ) from exc
    if not isinstance(document, dict):
        raise ValueError(f"stack_manifest: {registry_path} did not parse to a mapping")
    server_mode = document.get("server_mode")
    if not isinstance(server_mode, dict) or not server_mode:
        raise ValueError(
            f"stack_manifest: {registry_path} has no usable `server_mode:` section"
        )
    return document


_MANIFEST = _load_launch_manifest()
_MASTER = _load_master_registry()

# The master registry's per-server declarations, keyed by MASTER role name. Note
# that master and the launcher do not always use the same name for the same
# process (master says `worker`, the launcher says `worker_general`), which is
# what `master_server_row()` below exists to bridge.
MASTER_SERVER_MODE: dict[str, dict] = {
    name: cfg for name, cfg in _MASTER["server_mode"].items() if isinstance(cfg, dict)
}


def master_server_row(role: str) -> tuple[str | None, dict | None, str]:
    """Resolve a LAUNCHER role name to the master `server_mode` row that governs it.

    Returns ``(master_role_name, row, binding)``. The search order mirrors
    ``src/registry/stack_priors._server_for_role`` exactly — a divergence in the
    binding rules would mean the compiled priors and the launcher fallback
    disagreed about which declaration applies, which is the same bug one level up:
      1. a row named for the role itself,
      2. a row whose ``model_role`` is the role (master `worker` -> `worker_general`),
      3. a row whose ``shared_with`` contains the role (aliases).
    ``binding`` is "unresolved" when master says nothing about the role.
    """
    direct = MASTER_SERVER_MODE.get(role)
    if isinstance(direct, dict):
        return role, direct, "direct"
    for name, cfg in MASTER_SERVER_MODE.items():
        if cfg.get("model_role") == role:
            return name, cfg, "model_role"
        shared_with = cfg.get("shared_with")
        if isinstance(shared_with, list) and role in shared_with:
            return name, cfg, "shared_with"
    return None, None, "unresolved"


def _master_declared_model_path(role: str) -> str | None:
    """The GGUF path master declares for `role`, through the alias-aware binding.

    `model_path` is the explicit key; rows that carry only `model` spell the path
    there (as a string, or as `{"path": ...}` for the multimodal rows). A row that
    declares no path at all returns None -- "not declared" is not "missing file".
    """
    _name, row, _binding = master_server_row(role)
    if not isinstance(row, dict):
        return None
    declared = row.get("model_path")
    if not isinstance(declared, str):
        model = row.get("model")
        declared = model.get("path") if isinstance(model, dict) else model
    if isinstance(declared, str) and declared.strip().startswith("/"):
        return declared.strip()
    return None


# The master registry's KV-FEASIBILITY GROUP. `n_ctx`, `slots_by_shape` and
# `kv_quant` are declared together under `server_mode.<role>.serving_shape`
# because they are not independent: KV bytes are
# `KiB_per_token(model, kv_quant) * n_ctx`, and per-slot context is
# `n_ctx / slots`. See the long comment above `frontdoor:` in the master
# registry for the full argument.
SERVING_SHAPE_KEY = "serving_shape"


def master_serving_shape(role: str) -> tuple[dict, str | None]:
    """Master's ``serving_shape`` block for a LAUNCHER role name, if it has one."""
    name, row, binding = master_server_row(role)
    if row is None:
        return {}, None
    shape = row.get(SERVING_SHAPE_KEY)
    if not isinstance(shape, dict) or not shape:
        return {}, None
    return shape, f"{name}/{binding}.{SERVING_SHAPE_KEY}"


def master_declared(role: str, key: str) -> tuple[Any, str | None]:
    """Return ``(value, "<master_role>/<binding>")`` for a master-declared field.

    ``(None, None)`` when master declares nothing for that role/field.

    The grouped ``serving_shape`` block is consulted FIRST, then the flat key.
    Back-compat is a READ path, not a write path: a row that has not migrated
    (today only ``coder_escalation``, an alias with no serving shape of its own)
    still resolves through the flat key, but a row that HAS migrated must not
    keep a flat copy — ``validate_declaration_parity`` rejects that, because two
    live copies of one number is exactly what the grouping removes.
    """
    name, row, binding = master_server_row(role)
    if row is None:
        return None, None
    shape = row.get(SERVING_SHAPE_KEY)
    if isinstance(shape, dict) and shape.get(key) is not None:
        return shape[key], f"{name}/{binding}.{SERVING_SHAPE_KEY}"
    value = row.get(key)
    if value is None:
        return None, None
    return value, f"{name}/{binding}"


# =============================================================================
# Port assignments by role (primary ports — full-speed 1×96t instances)
# Pre-warm (2026-03-29): primary port is the full-speed instance.
# Quarter instances on offset ports (808x, 818x, 828x, 838x).
# Declared in launch_manifest.yaml `port_map:` — including the per-role comments
# that record WHY each port is what it is.
# =============================================================================

PORT_MAP = dict(_MANIFEST["port_map"])

# Stack-managed auxiliaries whose startup failure is non-fatal but must remain
# visible in the read-only status surface even when no state row was created.
OPTIONAL_AUXILIARY_ROLES = frozenset(_MANIFEST["optional_auxiliary_roles"])


# =============================================================================
# Aux services — stack-managed processes that are NOT llama-server roles
# Declared in launch_manifest.yaml `aux_services:`.
# =============================================================================


class AuxService(NamedTuple):
    """One declared auxiliary service.

    The registry this type populates is what makes `start` and `reload` agree.
    Before it existed, "things that can be started" was a set of bespoke
    `start_<name>()` functions and "things that can be reloaded" was an
    independent `elif` chain; whisper and sd_server were in the first and not the
    second, so reloading them killed the listener and then failed to bring it
    back. One table now feeds both.
    """

    name: str
    port: int
    argv: tuple[str, ...]
    cwd: str
    log: str
    model_label: str
    description: str = ""
    optional: bool = False
    backend: str | None = None
    env: dict[str, str] = {}  # noqa: RUF012 — replaced per-instance by _aux_service()
    pythonpath: tuple[str, ...] = ()
    ld_library_path: tuple[str, ...] = ()
    ld_library_path_mode: str = "prepend"
    verify_ggml_linkage: bool = False
    health_path: str = "/health"
    health_timeout: int = 60
    # OPTIONAL post-health smoke request: {path, method, json, min_bytes,
    # timeout}. `/health` proves the port answers; this proves the service
    # can produce OUTPUT. The TTS server returns HTTP 200 with zero audio
    # bytes for a request it cannot satisfy, which every status-code check
    # passes. Absent -> not checked.
    smoke: dict[str, Any] | None = None
    # OPTIONAL declared CPU placement (a `taskset -c` cpu list). Needed by any
    # CPU-resident ggml service whose threadpool is layout-sensitive: whisper.cpp
    # HANGS on some (threads, cpuset) pairs (epyc-inference-research
    # artifacts/speech_cpu_realtime_20260924/README.md, "Layout collapse"), so it
    # must run on a MEASURED-SAFE pair, never on whatever the bench guard picks.
    # Validated at LAUNCH (orchestrator_stack.start_aux_service), not at import:
    # a bad value must fail that one service, not every importer of this module.
    cpuset: str | None = None


_AUX_LD_MODES = ("prepend", "replace")


def _aux_service(entry: dict[str, Any]) -> AuxService:
    """Build one AuxService, failing loudly on a malformed declaration.

    No defaulting of required fields. A service whose declaration is incomplete
    must fail at import, not launch something plausible on the wrong port — the
    same contract `_load_launch_manifest` holds for the rest of this file.
    """
    for required in ("name", "port", "argv", "cwd", "log", "model_label"):
        if required not in entry:
            raise ValueError(f"aux_services: entry {entry.get('name', entry)!r} lacks {required!r}")
    backend = entry.get("backend")
    # A backend-resolved service is by definition running a foreign ggml
    # generation, so `replace` is its default: `prepend` would leave the ambient
    # llama.cpp tree reachable behind it, and a mixed-generation resolution does
    # not fail — it serves wrong answers.
    mode = entry.get("ld_library_path_mode", "replace" if backend else "prepend")
    if mode not in _AUX_LD_MODES:
        raise ValueError(
            f"aux_services: {entry['name']!r} declares ld_library_path_mode={mode!r}; "
            f"valid: {list(_AUX_LD_MODES)}"
        )
    return AuxService(
        name=str(entry["name"]),
        port=int(entry["port"]),
        argv=tuple(str(token) for token in entry["argv"]),
        cwd=str(entry["cwd"]),
        log=str(entry["log"]),
        model_label=str(entry["model_label"]),
        description=str(entry.get("description", "")),
        optional=bool(entry.get("optional", False)),
        backend=str(backend) if backend else None,
        env={str(key): str(value) for key, value in (entry.get("env") or {}).items()},
        pythonpath=tuple(str(path) for path in (entry.get("pythonpath") or ())),
        ld_library_path=tuple(str(path) for path in (entry.get("ld_library_path") or ())),
        ld_library_path_mode=mode,
        verify_ggml_linkage=bool(entry.get("verify_ggml_linkage", False)),
        health_path=str(entry.get("health_path", "/health")),
        health_timeout=int(entry.get("health_timeout", 60)),
        smoke=entry.get("smoke") if isinstance(entry.get("smoke"), dict) else None,
        cpuset=str(entry["cpuset"]).strip() if entry.get("cpuset") not in (None, "") else None,
    )


AUX_SERVICES: dict[str, AuxService] = {
    str(entry["name"]): _aux_service(entry) for entry in _MANIFEST["aux_services"]
}


def _validate_aux_services() -> None:
    """Import-time coherence checks for the aux-service registry.

    Both checks encode a defect that actually shipped:
      * a declared port that disagrees with PORT_MAP is how `reload <name>` and
        `status <name>` end up pointed at different listeners;
      * an optional auxiliary missing from PORT_MAP raises KeyError inside
        `cmd_status`'s unavailable-optional loop, i.e. the status surface dies
        precisely when a service is down and the operator needs to see it.
    """
    problems: list[str] = []
    for name, service in AUX_SERVICES.items():
        declared = PORT_MAP.get(name)
        if declared is not None and declared != service.port:
            problems.append(
                f"aux service {name!r}: PORT_MAP says {declared}, aux_services says {service.port}"
            )
        if service.optional and name not in PORT_MAP:
            problems.append(
                f"aux service {name!r} is optional but absent from PORT_MAP "
                f"(cmd_status indexes PORT_MAP for every optional auxiliary)"
            )
    for role in OPTIONAL_AUXILIARY_ROLES:
        if role not in PORT_MAP:
            problems.append(f"optional auxiliary role {role!r} is absent from PORT_MAP")
    if problems:
        raise ValueError("stack_manifest: aux service declaration is incoherent: " + "; ".join(problems))


_validate_aux_services()

# HOT roles (always started) - NUMA-optimized
HOT_ROLES = set(_MANIFEST["hot_roles"])

# All NUMA replica ports (for port scanning and cleanup) — derived from
# NUMA_CONFIG + PORT_MAP. Stays here because it depends on both the manifest's
# PORT_MAP and stack_numa.NUMA_CONFIG.
NUMA_REPLICA_PORTS = {
    port
    for cfg in NUMA_CONFIG.values()
    for _, port, _ in cfg["instances"]
    if port not in PORT_MAP.values()
}


# Roles that must never run concurrently (large/latency-sensitive paths).
# Note: frontdoor intentionally runs with 2 slots by default for better
# interactive responsiveness under concurrent traffic.
SERIAL_ROLES = set(_MANIFEST["serial_roles"])


# =============================================================================
# Single source of truth for "which roles run where" (refactored 2026-05-06)
# =============================================================================
# Pre-2026-05-06: HOT_SERVERS + WARM_SERVERS were two parallel hand-edited lists
# of dicts that duplicated wiring data already in NUMA_CONFIG. This made it easy
# to forget one when adding/removing/renaming a role and ship a broken config.
#
# Post-2026-05-06: HOT_SERVERS / WARM_SERVERS are COMPUTED from:
#   1. ROLE_LAUNCH_META — small classification dict: per-role tier + launch mode
#      + aliases + mode-specific kwargs. Declared in launch_manifest.yaml.
#   2. NUMA_CONFIG (stack_numa) — wiring spec: per-role NUMA instances list.
#      Declared in stack_topology.yaml.
#
# Adding a new role now requires editing TWO declared places consistently:
#   a) Add the role's NUMA wiring in stack_topology.yaml `numa_config:` (or set
#      `no_numa: true` in role_launch_meta if the role doesn't need NUMA pinning,
#      e.g. embedders).
#   b) Add a role_launch_meta entry with tier/mode/aliases.
# A consistency check (`_validate_role_classification()` below) catches
# common mismatches at module load time.

def _derived_aliases(role: str, meta: dict) -> list[str]:
    """Aliases that ride this role's process: launcher-only extras, then master's.

    Master's ``server_mode.<role>.shared_with`` is the declaration; the launcher
    used to keep a parallel ``shared_with_first_n`` list beside it, and the two
    drifted (master narrowed frontdoor's list at the W1 cutover while the
    launcher copy had to be edited separately in the same commit to keep up).

    ``launcher_only_aliases`` carries names master does not list — today only
    worker_explore, whose case is argued in launch_manifest.yaml. Extras LEAD so
    the emitted roles list keeps its historical order; the order is not
    functionally load-bearing (every alias resolves to the same port) but it
    surfaces in fleet markers and state keys, so it is worth not churning.
    """
    extras = [str(alias) for alias in meta.get("launcher_only_aliases") or []]
    shared_with, _source = master_declared(role, "shared_with")
    declared = [str(alias) for alias in shared_with or [] if isinstance(alias, str)]
    return extras + [alias for alias in declared if alias not in extras]


def _build_role_launch_meta() -> dict[str, dict]:
    """Per-role launch metadata, with the alias list DERIVED from master.

    Insertion order is load-bearing — HOT_SERVERS/WARM_SERVERS are emitted in it
    — and YAML mapping order is preserved on load.
    """
    out: dict[str, dict] = {}
    for role, declared_meta in _MANIFEST["role_launch_meta"].items():
        meta = dict(declared_meta)
        if "shared_with_first_n" in meta:
            raise ValueError(
                f"stack_manifest: role_launch_meta['{role}'] declares "
                f"shared_with_first_n, which phase 2 DERIVES from "
                f"server_mode.{role}.shared_with. Add the alias to the master "
                f"registry, or to launcher_only_aliases with a parity exception."
            )
        aliases = _derived_aliases(role, meta)
        meta.pop("launcher_only_aliases", None)
        if aliases:
            # Only set the key when non-empty: an empty list is a different dict
            # shape from an absent key, and consumers iterate this dict.
            meta["shared_with_first_n"] = aliases
        out[role] = meta
    return out


ROLE_LAUNCH_META: dict[str, dict] = _build_role_launch_meta()



# =============================================================================
# Model paths — declared in launch_manifest.yaml `models:` / `embedding:` /
# `vision:`. The measured provenance for each choice moved with the data.
# =============================================================================

_MODELS = _MANIFEST["models"]
_EMBEDDING = _MANIFEST["embedding"]

# Embedding model: BGE-large-en-v1.5 (purpose-built for embeddings, 1024 dims)
# 6 parallel instances provide redundancy and reduce latency via fan-out
EMBEDDING_MODEL_PATH = _MODELS["embedding_model_path"]
EMBEDDER_PORTS = list(_EMBEDDING["pool_ports"])
# Each pool port gets its OWN recipe dict (never a shared one — a caller that
# mutated a shared dict would silently retune the whole pool).
EMBEDDING_SERVER_RECIPES: dict[int, dict[str, str | int | bool]] = {
    port: {"model_path": EMBEDDING_MODEL_PATH, **_EMBEDDING["pool_recipe"]}
    for port in EMBEDDER_PORTS
}
EMBEDDING_SERVER_RECIPES.update(
    {int(port): dict(recipe) for port, recipe in _EMBEDDING["extra_recipes"].items()}
)


# ── Embedder placement (UFH-12 Phase 0, 2026-09-26) ─────────────────────────
# Declared per port in launch_manifest.yaml `embedding.placement`. Until this
# existed the pool had no placement at all and the canonical OMP env bound all
# six servers' compute threads to the same four core places. The NUMA node is
# DERIVED from the cpuset (never declared beside it), and every invariant below
# is recomputed from the declarations, so none of them can drift from a copy.
class EmbedderPlacement(NamedTuple):
    cpuset: str
    numa_node: int


_EMBED_CPUSET_RE = re.compile(r"\d+(-\d+)?(,\d+(-\d+)?)*")
_HOST_LOGICAL_CPUS = 192
_HOST_PHYSICAL_CORES = 96


def _smt_fold(cpus: set[int]) -> set[int]:
    """Logical cpus -> the physical cores they run on (sibling of c is c +/- 96)."""
    return {c % _HOST_PHYSICAL_CORES for c in cpus}


def _load_embedding_placement(
    declared: Any,
    *,
    recipes: dict[int, dict[str, Any]],
    pool_ports: list[int],
    aux_services: dict[str, "AuxService"],
) -> dict[int, EmbedderPlacement]:
    """Validate `embedding.placement` and return {port: EmbedderPlacement}.

    Raises ValueError listing EVERY violation at once:
      * every pool port is declared (an undeclared pool port is the old defect);
      * a declared port has a recipe, and its cpuset is a cpu list on this host;
      * the cpuset lies in exactly ONE NPS4 node (membind target is derived);
      * the recipe's `threads` equals the cpuset's logical-cpu count;
      * cpusets are pairwise disjoint AFTER SMT folding (two instances on the
        two siblings of one core would contend exactly as the old layout did);
      * no cpuset SMT-folds onto an aux service's declared cpuset (whisper.cpp
        hangs on unmeasured layouts) or onto the GPU host lane.
    """
    if declared is None:
        declared = {}
    if not isinstance(declared, dict):
        raise ValueError("launch_manifest: embedding.placement must be a mapping of port -> {cpuset}")
    problems: list[str] = []
    out: dict[int, EmbedderPlacement] = {}
    folded_by_port: dict[int, set[int]] = {}
    for raw_port, entry in declared.items():
        port = int(raw_port)
        if port not in recipes:
            problems.append(f":{port} has a placement but no embedding recipe")
            continue
        if not isinstance(entry, dict) or set(entry) != {"cpuset"}:
            problems.append(f":{port} placement must be exactly {{cpuset: <cpu list>}}, got {entry!r}")
            continue
        cpuset = str(entry["cpuset"]).strip()
        if not _EMBED_CPUSET_RE.fullmatch(cpuset):
            problems.append(f":{port} cpuset {cpuset!r} is not a cpu list")
            continue
        cpus = _parse_cpus(cpuset)
        if max(cpus) >= _HOST_LOGICAL_CPUS:
            problems.append(f":{port} cpuset {cpuset!r} names a cpu beyond 0-{_HOST_LOGICAL_CPUS - 1}")
            continue
        nodes = _nodes_touched(cpuset)
        if len(nodes) != 1:
            problems.append(f":{port} cpuset {cpuset!r} spans NPS4 nodes {nodes}; an embedder must sit in one")
            continue
        threads = int(recipes[port].get("threads", 0))
        if threads != len(cpus):
            problems.append(
                f":{port} recipe -t {threads} but cpuset {cpuset!r} holds {len(cpus)} logical cpus"
            )
        folded_by_port[port] = _smt_fold(cpus)
        out[port] = EmbedderPlacement(cpuset=cpuset, numa_node=nodes[0])
    missing = sorted(set(pool_ports) - set(out) - {int(p) for p in declared})
    if missing:
        problems.append(f"pool ports without a declared placement: {missing}")
    ports = sorted(folded_by_port)
    for i, a in enumerate(ports):
        for b in ports[i + 1:]:
            shared = folded_by_port[a] & folded_by_port[b]
            if shared:
                problems.append(f":{a} and :{b} share physical cores {sorted(shared)} after SMT folding")
    fenced = {"GPU host lane": GPU_HOST_LANE[0]}
    fenced.update({f"aux {name}": svc.cpuset for name, svc in aux_services.items() if svc.cpuset})
    for label, spec in fenced.items():
        fence = _smt_fold(_parse_cpus(spec))
        for port in ports:
            shared = folded_by_port[port] & fence
            if shared:
                problems.append(f":{port} SMT-folds onto the {label} ({spec}) at cores {sorted(shared)}")
    if problems:
        raise ValueError(
            "launch_manifest: embedding.placement is incoherent:\n  " + "\n  ".join(problems)
        )
    return out


EMBEDDING_PLACEMENT: dict[int, EmbedderPlacement] = _load_embedding_placement(
    _EMBEDDING.get("placement"),
    recipes=EMBEDDING_SERVER_RECIPES,
    pool_ports=EMBEDDER_PORTS,
    aux_services=AUX_SERVICES,
)

# Worker pool models (FIXED paths to existing files)
# NOTE: worker_coder uses the fast 1.5B worker backend on port 8102.
WORKER_POOL_MODELS = dict(_MODELS["worker_pool"])

# Draft model for MTP speculative decoding on explore worker.
EXPLORE_DRAFT_MODEL = _MODELS["explore_draft_model"]

# Vision models (VL) with multimodal projector.
_VISION_WORKER = _MANIFEST["vision"]["worker"]
_VISION_ESCALATION = _MANIFEST["vision"]["escalation"]
VISION_WORKER_MODEL = _VISION_WORKER["model"]
VISION_WORKER_MMPROJ = _VISION_WORKER["mmproj"]
# DERIVED (phase 2) from server_mode.worker_vision.device. The launcher used to
# declare `device: ROCm0` in vision.worker; that second copy is exactly the
# surface the `--device none` incident ran through, and master also declares
# the 27B's device (architect_general until the 2026-09-27 ARCHITECT SWAP,
# architect_critic since) — a role this block could not name at all.
_VISION_WORKER_DEVICE, _VISION_WORKER_DEVICE_SOURCE = master_declared("worker_vision", "device")
if not isinstance(_VISION_WORKER_DEVICE, str) or not _VISION_WORKER_DEVICE:
    raise ValueError(
        "stack_manifest: server_mode.worker_vision.device is not declared in the master "
        "registry. The launcher derives the VL processor from it and keeps no local copy; "
        "declare it in orchestration/model_registry.yaml rather than re-adding one here."
    )
if "device" in _VISION_WORKER:
    raise ValueError(
        "stack_manifest: launch_manifest.yaml vision.worker re-declares `device`, which "
        "phase 2 derives from server_mode.worker_vision.device. Two declarations is how "
        "the GPU role ended up on CPU; remove it and change the master registry instead."
    )
VISION_WORKER_DEVICE = _VISION_WORKER_DEVICE
VISION_WORKER_IMAGE_MIN_TOKENS = _VISION_WORKER["image_min_tokens"]
VISION_WORKER_CACHE_RAM = _VISION_WORKER["cache_ram"]
# vision_escalation is an ALIAS on the worker's process, not a second server, so
# its model/mmproj/device are BOUND rather than declared twice. Declaring them
# separately is exactly how the retired :8087 lane drifted to a different model.
if not _VISION_ESCALATION.get("same_as_worker"):
    raise ValueError(
        "stack_manifest: vision.escalation must declare same_as_worker: true — "
        "vision_escalation is an alias on the worker_vision process, and a second "
        "independent model declaration is the drift this binding exists to prevent"
    )
VISION_ESCALATION_MODEL = VISION_WORKER_MODEL
VISION_ESCALATION_MMPROJ = VISION_WORKER_MMPROJ
VISION_ESCALATION_DEVICE = VISION_WORKER_DEVICE
VISION_ESCALATION_REASONING = _VISION_ESCALATION["reasoning"]

_LAUNCH_SHAPE = _MANIFEST["launch_shape"]
# Host-side capacity policy for the serving_shape capacity guard. The GPU side
# needs nothing here — device_model owns the declared VRAM total and headroom.
_CAPACITY = _MANIFEST["capacity"]
DEFAULT_EFFECTIVE_CONTEXT_TOKENS = _LAUNCH_SHAPE["default_effective_context_tokens"]

DEFAULT_UBATCH_TOKENS = _LAUNCH_SHAPE["default_ubatch_tokens"]


def _launcher_role_universe() -> list[str]:
    """Every role name the launcher can be asked about — primaries and aliases.

    Aliases are included because an alias rides its host's process and must
    resolve to the SAME serving shape; a table that covered only primaries would
    make `LAUNCH_CONTEXT_TOKENS["vision_escalation"]` a KeyError in the escalation
    branch of the VL builder.
    """
    roles: set[str] = set(_MANIFEST["port_map"]) | set(_MANIFEST["role_launch_meta"])
    roles.update(NUMA_CONFIG)
    for meta in ROLE_LAUNCH_META.values():
        roles.update(str(alias) for alias in meta.get("shared_with_first_n") or [])
    return sorted(roles)


def _derived_context_tokens() -> dict[str, int]:
    """`-c` per role: master's `serving_shape.n_ctx` first, manifest rows second.

    2026-08-02: `n_ctx` used to be declared on BOTH sides and merely cross-checked
    (`parity.checked`). It is now DERIVED, like `slots` and `shared_with` before
    it, because it is one third of the KV-feasibility group and a launcher-side
    copy could be edited without master's capacity check ever seeing it. The
    manifest keeps only rows master has no row for — today just `worker_fast`.
    """
    out: dict[str, int] = {
        str(role): int(value)
        for role, value in (_LAUNCH_SHAPE.get("context_tokens") or {}).items()
    }
    for role in _launcher_role_universe():
        declared, _source = master_declared(role, "n_ctx")
        if isinstance(declared, int) and not isinstance(declared, bool) and declared > 0:
            out[role] = declared
    return out


def _derived_kv_quant_configs() -> dict[str, tuple[str, str]]:
    """`-ctk`/`-ctv` per role, DERIVED from master's `serving_shape.kv_quant`.

    2026-08-02: `launch_shape.kv_quant_configs` is DELETED. It held 9 rows while
    master held 2, so the launcher looked authoritative and master looked
    incomplete — but master is where the KV type belongs, since the KV footprint
    it implies is only meaningful next to the `n_ctx` it multiplies. All 7 rows
    master was missing were verified against the LIVE `/proc/<pid>/cmdline`
    `-ctk`/`-ctv` before being written there, so this move recorded reality
    rather than changing it.

    A role with no declaration gets no entry and therefore no `-ctk`/`-ctv` at
    all — today that is the embedders and `worker_fast`, which declare no
    `serving_shape`.

    `worker_vision` USED to be the example here, on the grounds that it "keeps
    llama-server's f16 default while a q8_0 VL quality check runs separately".
    That check has since run and landed: the MMMU-250 paired A/B (f16 153/250 vs
    q8_0 155/250, +0.80 pp, CI [-1.90,+3.03], non-inferior at a pre-registered
    3 pp margin) moved `server_mode.worker_vision.serving_shape.kv_quant` to
    q8_0/q8_0, so the VL roles are now covered like any other. Do not read this
    resolver as a vision exemption — `_build_vision_command` emits whatever
    master declares, same as every other builder.

    The resolver is per-ROLE today. It reads through `master_declared`, which
    already consults `serving_shape` first, so a future `kv_quant_by_shape` is a
    schema addition here and in `resolve_slots` — not a migration.
    """
    out: dict[str, tuple[str, str]] = {}
    for role in _launcher_role_universe():
        raw, _source = master_declared(role, "kv_quant")
        if isinstance(raw, dict) and raw.get("k") and raw.get("v"):
            out[role] = (str(raw["k"]), str(raw["v"]))
    return out


LAUNCH_CONTEXT_TOKENS = _derived_context_tokens()

# Effective launcher KV settings. Descriptors may preserve broader model
# capability metadata; this table witnesses the actual llama-server CLI path.
LAUNCH_KV_QUANT_CONFIGS = _derived_kv_quant_configs()
NO_SPEC_DECODE_ROLES: set[str] = set(_LAUNCH_SHAPE["no_spec_decode_roles"])
KV_HADAMARD_ROLES = set(_V2_ROLES)


# =============================================================================
# Slots (-np) — DERIVED from the master registry
# =============================================================================
# `slots` is how many concurrent slots the SERVER is launched with. It is NOT
# SERIAL_ROLES, which is an admission policy (how many requests the orchestrator
# will have in flight for a role). The launcher computed `-np` as
# `1 if role in SERIAL_ROLES else 2`, which conflated the two and disagreed with
# the declared `server_mode.<role>.slots` for 6 of the 11 role names master
# declares it for. A role may legitimately run a 2-slot server while admission
# serialises it; the two must be derived from two different declarations.

FALLBACK_SLOTS: dict[str, int] = {
    key: int(value) for key, value in _LAUNCH_SHAPE["fallback_slots"].items()
}


class SlotsDecision(NamedTuple):
    """A resolved slot count and WHERE it came from.

    The source string is not decoration. A config surface that hands back
    fallbacks without saying they are fallbacks makes "master declares 1 and we
    launched 2" indistinguishable from "nobody declared anything" — which is how
    the vision defect stayed invisible while the compiled artifact was correct.
    """

    slots: int
    source: str

    @property
    def declared(self) -> bool:
        return self.source.startswith("master:")


def _fallback_slots_key(mode: str, worker_type: str | None, vision_type: str | None) -> str:
    """Manifest key for a launch mode's fallback, most specific first."""
    if mode == "worker_pool" and worker_type:
        specific = f"worker_pool_{worker_type}"
        if specific in FALLBACK_SLOTS:
            return specific
    if mode == "vision" and vision_type:
        specific = f"vision_{vision_type}"
        if specific in FALLBACK_SLOTS:
            return specific
    return mode if mode in FALLBACK_SLOTS else "default"


def fallback_slots_for_mode(
    mode: str = "default",
    *,
    worker_type: str | None = None,
    vision_type: str | None = None,
) -> SlotsDecision:
    """The launcher's declared per-mode slot default. Consults NO registry.

    For callers that already resolve the declaration themselves against a
    registry they were HANDED — the stack-prior compiler takes a
    ``registry_path`` and must honour that one, not whatever registry this
    module happened to load. Reading master here as well would give the compiler
    two sources for one field, which is the bug one level up.
    """
    key = _fallback_slots_key(mode, worker_type, vision_type)
    return SlotsDecision(FALLBACK_SLOTS[key], f"manifest:fallback_slots.{key}")


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def slots_by_shape_for(role: str) -> tuple[dict[str, int], str | None]:
    """Master's per-shape-class slot map for a role, if it declares one.

    Keys are SHAPE CLASSES (`full` / `half` / `quarter` / `gpu_host_lane`), never
    cpusets — see `stack_numa._SHAPE_CLASSES`.
    """
    raw, source = master_declared(role, "slots_by_shape")
    if not isinstance(raw, dict):
        return {}, None
    table = {
        str(cls): parsed
        for cls, value in raw.items()
        if (parsed := _positive_int(value)) is not None
    }
    return table, source


def resolve_slots(
    role: str,
    mode: str = "default",
    *,
    worker_type: str | None = None,
    vision_type: str | None = None,
    numa_instance: int | None = None,
) -> SlotsDecision:
    """Resolve `-np` for one INSTANCE of a role.

    Precedence, most specific first:
      1. ``serving_shape.slots_by_shape[<class of this instance>]`` — the
         per-instance answer. Requires ``numa_instance``; the instance's shape
         CLASS comes from ``stack_topology.yaml``'s declared ``cpu_shape``, so no
         role -> shape table exists anywhere.
      2. the role's flat ``slots`` — the compat scalar, and the whole answer for
         a role with one instance class (both GPU roles).
      3. the manifest's per-mode fallback, for a role master says nothing about.

    2026-08-02: (1) is new. `slots` is per-ROLE, but the operator's spec is per
    INSTANCE SHAPE — frontdoor's 96-core full wants 16 slots while each 48-core
    half wants 4, and one number cannot say that. A class the role does not
    declare falls through to (2) rather than erroring, which is what makes
    ``slots_by_shape`` optional and keeps the GPU roles on a single scalar.

    SERIAL_ROLES deliberately does NOT appear here. Serialising a role is an
    admission decision and it is made in src/api/admission.py; expressing it by
    shrinking the server's slot count instead both loses the distinction and
    silently overrides a declaration.
    """
    if numa_instance is not None:
        shape_class = instance_shape_class(role, numa_instance)
        if shape_class is not None:
            table, source = slots_by_shape_for(role)
            if shape_class in table:
                return SlotsDecision(
                    table[shape_class], f"master:{source}.{shape_class}"
                )

    declared, source = master_declared(role, "slots")
    parsed = _positive_int(declared)
    if parsed is not None:
        return SlotsDecision(parsed, f"master:{source}")
    return fallback_slots_for_mode(mode, worker_type=worker_type, vision_type=vision_type)


def _declared_slots_table() -> dict[str, int]:
    """Every launcher role name master declares a slot count for."""
    table: dict[str, int] = {}
    for role in sorted(set(_MANIFEST["port_map"]) | set(_MANIFEST["role_launch_meta"])):
        declared, _source = master_declared(role, "slots")
        parsed = _positive_int(declared)
        if parsed is not None:
            table[role] = parsed
    return table


def declared_slots_by_port(role: str) -> dict[int, int]:
    """Port -> resolved `-np`, for every declared instance of a role.

    The JOIN, at its smallest: `stack_topology.yaml` supplies the ports and the
    shape classes, the master registry supplies the slots per class. Used by the
    parity guard and the capacity check; the compiled priors carry the same map
    for the launcher and for admission, resolved against the registry the
    COMPILER was handed rather than the ambient one.
    """
    cfg = NUMA_CONFIG.get(role)
    if not cfg:
        return {}
    out: dict[int, int] = {}
    for idx, instance in enumerate(cfg["instances"]):
        out[int(instance[1])] = resolve_slots(role, numa_instance=idx).slots
    return out


# Role name -> declared slot count, for inspection/reporting and the parity
# guard. Absent keys are roles master says nothing about.
DECLARED_SLOTS: dict[str, int] = _declared_slots_table()

# =============================================================================
# GPU shadow lane launch constants (docs/gpu-shadow-lane.md; gpu-serving-tie-in
# P2-6/P0-2). Declared in launch_manifest.yaml `gpu_shadow_lane:`; INERT until a
# registry proposal adds the matching role_launch_meta + numa_config + port_map
# entries.
# =============================================================================

_GPU_SHADOW_LANE = _MANIFEST["gpu_shadow_lane"]
GPU_SHADOW_LANE_TENANT_ROLE = _GPU_SHADOW_LANE["tenant_role"]
GPU_SHADOW_LANE_DEVICE = _GPU_SHADOW_LANE["device"]
GPU_SHADOW_LANE_REASONING = _GPU_SHADOW_LANE["reasoning"]
GPU_SHADOW_LANE_FALLBACK_SLOTS = _GPU_SHADOW_LANE["fallback_slots"]
GPU_SHADOW_LANE_FALLBACK_CONTEXT_TOKENS = _GPU_SHADOW_LANE["fallback_context_tokens"]

DEV_MODEL = _MODELS["dev_model"]
DEV_MODEL_PATH = str(_PATHS["models_dir"] / DEV_MODEL)

# Optional orchestrator API launch profiles for repeatable debugging runs.
ORCHESTRATOR_PROFILES: dict[str, dict[str, str]] = {
    name: dict(env) for name, env in _MANIFEST["orchestrator_profiles"].items()
}


# =============================================================================
# Docker Services (NextPLAID retrieval + SearXNG metasearch + Crawl4AI extraction)
# Declared in launch_manifest.yaml `docker_services:`, including the port-choice
# rationale (searxng 8888 not 8090; crawl4ai 11235 not 8086).
# =============================================================================

DOCKER_SERVICES = [dict(service) for service in _MANIFEST["docker_services"]]



# =============================================================================
# Server-list classification helpers
# =============================================================================


def _filter_by_numa_mode(servers: list[dict], mode: str) -> list[dict]:
    """Filter server list by --numa-mode {full,quarter,both}.

    For roles whose NUMA_CONFIG has both `full_instance_idx` AND multiple
    instances (i.e., a full-NUMA-node instance + per-half-NUMA-node siblings
    on overlapping CPU sets), pick one mode:
      - "full":    keep only the full instance (max single-stream tps)
      - "quarter": skip the full, keep the sub-full siblings. The token is
                   legacy: since the 2026-07-30 quarter retirement those
                   siblings are HALF instances (2x48t), not quarters (4x24t).
      - "both":    return input unchanged — full + half siblings, the ratified
                   production mode (stack_topology.yaml `numa_mode: both`)

    Roles without a full+sub-full mix (single-instance roles like
    architect_general, or sub-full-only roles like vision_escalation) pass
    through untouched.
    """
    if mode == "both":
        return servers
    out: list[dict] = []
    dropped_full_aliases: dict[str, list[str]] = {}
    first_kept_by_role: dict[str, dict] = {}
    for srv in servers:
        # The primary role is roles[0]; aliases share its NUMA_CONFIG.
        roles = srv["roles"]
        role = roles[0]
        cfg = NUMA_CONFIG.get(role)
        if not cfg or "full_instance_idx" not in cfg or len(cfg["instances"]) <= 1:
            out.append(srv)
            continue
        full_idx = cfg["full_instance_idx"]
        srv_idx = srv.get("numa_instance", 0)
        if mode == "full" and srv_idx == full_idx:
            out.append(srv)
        elif mode == "quarter" and srv_idx != full_idx:
            kept = dict(srv)
            kept["roles"] = list(roles)
            out.append(kept)
            first_kept_by_role.setdefault(role, kept)
        elif mode == "quarter" and srv_idx == full_idx and len(roles) > 1:
            dropped_full_aliases.setdefault(role, []).extend(str(alias) for alias in roles[1:])
    if mode == "quarter":
        for role, aliases in dropped_full_aliases.items():
            target = first_kept_by_role.get(role)
            if not target:
                continue
            roles = target.get("roles")
            if not isinstance(roles, list):
                continue
            for alias in aliases:
                if alias not in roles:
                    roles.append(alias)
    return out


def _build_servers_from_classification() -> tuple[list[dict], list[dict]]:
    """Compute HOT_SERVERS + WARM_SERVERS from ROLE_LAUNCH_META + NUMA_CONFIG.

    For each role in ROLE_LAUNCH_META:
      - If "no_numa": True, emit a single server entry at meta["port"].
      - Else look up NUMA_CONFIG[role]["instances"] and emit one entry per instance.
    Mode-specific flags (vision/worker_pool/embedding) get added to each entry,
    plus mode-specific kwargs (vision_type, worker_type).
    Aliases (shared_with_first_n) are added to the first N entries only.
    """
    hot: list[dict] = []
    warm: list[dict] = []

    for role, meta in ROLE_LAUNCH_META.items():
        target = hot if meta["tier"] == "hot" else warm
        mode = meta.get("mode", "default")
        aliases = meta.get("shared_with_first_n", [])
        alias_count = meta.get(
            "shared_with_first_n_count", 1
        )  # default: aliases on first instance only

        # Mode-specific flag dict applied to every entry for this role
        mode_flags: dict = {}
        if mode == "worker_pool":
            mode_flags["worker_pool"] = True
            if "worker_type" in meta:
                mode_flags["worker_type"] = meta["worker_type"]
        elif mode == "vision":
            mode_flags["vision"] = True
            if "vision_type" in meta:
                mode_flags["vision_type"] = meta["vision_type"]
        elif mode == "embedding":
            mode_flags["embedding"] = True
        elif mode == "eval_batch_frontdoor":
            mode_flags["eval_batch_frontdoor"] = True
        elif mode == "gpu_shadow_lane":
            # gpu-serving-tie-in P2-6 (P0-2): mode branch for the GPU shadow
            # lane launcher (docs/gpu-shadow-lane.md). INERT today — no
            # ROLE_LAUNCH_META entry carries this mode until the registry
            # proposal (docs/proposals/gpu-shadow-lane-registry-proposal.md)
            # is applied at the operator-gated activation. Witnessed by
            # tests/unit/test_gpu_shadow_lane.py (State-A inertness witness).
            mode_flags["gpu_shadow_lane"] = True

        if meta.get("no_numa"):
            # Single port, no NUMA pinning (embedders, worker_fast, etc.)
            entry = {"port": meta["port"], "roles": [role]}
            entry.update(mode_flags)
            target.append(entry)
            continue

        # NUMA-pinned: one entry per instance in NUMA_CONFIG[role]["instances"]
        cfg = NUMA_CONFIG.get(role)
        if not cfg:
            raise ValueError(
                f"Role '{role}' in ROLE_LAUNCH_META requires NUMA_CONFIG entry "
                f"(or set 'no_numa': True if no pinning needed)"
            )
        for idx, instance in enumerate(cfg["instances"]):
            _cpus, port, _threads = instance
            roles_list: list[str] = [role]
            if idx < alias_count:
                roles_list.extend(aliases)
            entry = {"port": port, "roles": roles_list}
            if "full_instance_idx" in cfg or len(cfg["instances"]) > 1:
                entry["numa_instance"] = idx
            entry.update(mode_flags)
            target.append(entry)

    return hot, warm


def _validate_role_classification() -> None:
    """Sanity checks on ROLE_LAUNCH_META + NUMA_CONFIG agreement.

    Catches the common bugs that hand-edited HOT_SERVERS used to allow:
      - Role in ROLE_LAUNCH_META but missing NUMA_CONFIG (and no "no_numa" flag)
      - Role in NUMA_CONFIG but missing ROLE_LAUNCH_META classification
      - Same port assigned to two different role primaries
    Raises ValueError on any inconsistency.
    """
    errors: list[str] = []

    # Check 1: every NUMA_CONFIG role has a ROLE_LAUNCH_META entry
    for role in NUMA_CONFIG:
        if role not in ROLE_LAUNCH_META:
            errors.append(
                f"NUMA_CONFIG['{role}'] has no matching ROLE_LAUNCH_META entry — "
                f"add tier/mode classification or remove from NUMA_CONFIG"
            )

    # Check 2: every NUMA-pinned ROLE_LAUNCH_META role has NUMA_CONFIG instances
    for role, meta in ROLE_LAUNCH_META.items():
        if meta.get("no_numa"):
            if "port" not in meta:
                errors.append(f"ROLE_LAUNCH_META['{role}'] no_numa=True requires 'port' field")
        else:
            if role not in NUMA_CONFIG:
                errors.append(
                    f"ROLE_LAUNCH_META['{role}'] has no_numa=False but no NUMA_CONFIG entry"
                )

    # Check 3: no port collisions between role primaries
    primary_ports: dict[int, str] = {}
    for role, meta in ROLE_LAUNCH_META.items():
        if meta.get("no_numa"):
            ports = [meta["port"]]
        else:
            cfg = NUMA_CONFIG.get(role, {})
            ports = [inst[1] for inst in cfg.get("instances", [])]
        for port in ports:
            if port in primary_ports and primary_ports[port] != role:
                errors.append(f"Port {port} assigned to both '{primary_ports[port]}' and '{role}'")
            primary_ports[port] = role

    if errors:
        msg = "Role classification inconsistencies detected:\n  " + "\n  ".join(errors)
        raise ValueError(msg)


# =============================================================================
# Parity guard — no field may be declared twice and disagree
# =============================================================================
# `_validate_role_classification()` above checks the launcher against ITSELF
# (manifest vs topology). This one checks the launcher against the MASTER
# registry, which is the direction every severe launcher finding came from: the
# compiled artifact was right and a local copy won anyway.
#
# It is not a warning list. `validate_against_registry()` already returns
# warnings and is called non-fatally from `start`; warnings are how a
# divergence survives for three months. This raises, at import, before anything
# is launched.

_PARITY = _MANIFEST["parity"]


def _parity_exception_keys() -> set[tuple[str, str]]:
    """(field, launcher_role) pairs the manifest argues for keeping divergent."""
    keys: set[tuple[str, str]] = set()
    for entry in _PARITY.get("exceptions") or []:
        if not isinstance(entry, dict):
            continue
        field = entry.get("field")
        role = entry.get("launcher_role")
        if isinstance(field, str) and isinstance(role, str):
            if not str(entry.get("reason") or "").strip():
                raise ValueError(
                    f"stack_manifest: parity exception {field}/{role} has no reason. "
                    f"An exception without an argument is just a silence with extra steps."
                )
            keys.add((field, role))
    return keys


def _launcher_declarations(field: str) -> dict[str, object]:
    """The launcher's rows for a doubly-declared field, keyed by ROLE name."""
    if field == "port":
        return dict(PORT_MAP)
    if field == "tier":
        return {role: meta.get("tier") for role, meta in ROLE_LAUNCH_META.items()}
    if field == "kv_quant":
        return {role: tuple(pair) for role, pair in LAUNCH_KV_QUANT_CONFIGS.items()}
    if field == "n_ctx":
        return dict(LAUNCH_CONTEXT_TOKENS)
    if field in ("numa_ports", "numa_instances"):
        # master's `numa_ports` is the launcher's instance list MINUS the full
        # instance (a role with no `full_instance_idx` has no full instance to
        # remove), and `numa_instances` is its length. stack_topology declares
        # (cpus, port, threads) per instance, which master cannot express, so the
        # launcher holds the richer declaration and master holds a projection of
        # it — this reconstructs the projection and compares that.
        rows: dict[str, object] = {}
        for role, cfg in NUMA_CONFIG.items():
            instances = cfg.get("instances") or []
            full_idx = cfg.get("full_instance_idx")
            ports = [
                instance[1]
                for idx, instance in enumerate(instances)
                if idx != full_idx or full_idx is None
            ]
            rows[role] = ports if field == "numa_ports" else len(ports)
        return rows
    raise ValueError(f"stack_manifest: parity guard has no launcher accessor for {field!r}")


def _master_declaration(role: str, field: str) -> tuple[object, str | None]:
    """Master's value for a doubly-declared field, in the launcher's units."""
    if field == "kv_quant":
        raw, source = master_declared(role, "kv_quant")
        if isinstance(raw, dict) and raw.get("k") and raw.get("v"):
            return (str(raw["k"]), str(raw["v"])), source
        return None, None
    return master_declared(role, field)


def validate_declaration_parity() -> None:
    """Fail if any field declared in BOTH this launcher and master disagrees.

    Two checks:
      1. DERIVED fields must not be re-declared launcher-side. A derivation dies
         when somebody adds the value back "for clarity"; the guard is the thing
         that notices.
      2. Every field in `parity.checked` is intersected on role name and
         compared. Rows only one side declares are fine — they are why the field
         is still declared here at all. Rows both declare and disagree are not.
    """
    errors: list[str] = []
    exceptions = _parity_exception_keys()
    used_exceptions: set[tuple[str, str]] = set()

    # --- 1. derived fields must have no launcher-side declaration -------------
    derived = set(_PARITY.get("derived_from_master") or [])
    if "slots" in derived:
        for role, meta in _MANIFEST["role_launch_meta"].items():
            if "slots" in meta:
                errors.append(
                    f"role_launch_meta['{role}'] declares `slots`, which is DERIVED "
                    f"from server_mode.{role}.slots"
                )
        if "slots" in _LAUNCH_SHAPE:
            errors.append("launch_shape declares a `slots` table, which is DERIVED from master")
    if "kv_quant" in derived and "kv_quant_configs" in _LAUNCH_SHAPE:
        errors.append(
            "launch_shape declares `kv_quant_configs`, which is DERIVED from "
            "server_mode.<role>.serving_shape.kv_quant. It was DELETED on 2026-08-02 "
            "after all 9 of its rows were verified against the live -ctk/-ctv and "
            "written to master; re-adding it recreates the 9-vs-2 split where the "
            "launcher outranked the registry on a model fact."
        )
    if "n_ctx" in derived:
        for role in sorted(_LAUNCH_SHAPE.get("context_tokens") or {}):
            declared, source = master_declared(str(role), "n_ctx")
            if declared is not None:
                errors.append(
                    f"launch_shape.context_tokens['{role}'] re-declares `n_ctx`, which is "
                    f"DERIVED from master ({source}, {declared!r}). n_ctx is one third of "
                    f"the KV-feasibility group; a launcher-side copy can be edited without "
                    f"the capacity check ever seeing it."
                )
    if "device" in derived and "device" in _MANIFEST["vision"]["worker"]:
        errors.append("vision.worker declares `device`, which is DERIVED from master")
    if "shared_with" in derived:
        for role, meta in _MANIFEST["role_launch_meta"].items():
            if "shared_with_first_n" in meta:
                errors.append(
                    f"role_launch_meta['{role}'] declares `shared_with_first_n`, which is "
                    f"DERIVED from server_mode.{role}.shared_with"
                )

    # --- 2. doubly-declared fields must agree row by row ----------------------
    for field in _PARITY.get("checked") or []:
        launcher_rows = _launcher_declarations(str(field))
        for role, launcher_value in sorted(launcher_rows.items()):
            if launcher_value is None:
                continue
            master_value, source = _master_declaration(role, str(field))
            if master_value is None:
                continue
            if isinstance(master_value, list):
                master_value = list(master_value)
            if isinstance(launcher_value, list):
                launcher_value = list(launcher_value)
            if launcher_value == master_value:
                continue
            key = (str(field), role)
            if key in exceptions:
                used_exceptions.add(key)
                continue
            errors.append(
                f"{field} for role '{role}': launcher declares {launcher_value!r}, "
                f"master ({source}) declares {master_value!r}"
            )

    # --- 3. the derived alias lists must still match master ------------------
    # `shared_with` is derived, so it cannot disagree by construction — but the
    # launcher-only extras CAN turn into permanent shadow declarations. Assert
    # each extra is covered by an argued exception, and drop the exception the
    # moment master starts declaring it.
    for role, meta in _MANIFEST["role_launch_meta"].items():
        extras = meta.get("launcher_only_aliases") or []
        if not extras:
            continue
        key = ("shared_with", role)
        declared, source = master_declared(role, "shared_with")
        already = [alias for alias in extras if alias in (declared or [])]
        if already:
            errors.append(
                f"role_launch_meta['{role}'].launcher_only_aliases lists {already!r}, "
                f"which master ({source}) now declares too — delete the extra and the "
                f"parity exception; it is no longer launcher-only"
            )
            continue
        if key not in exceptions:
            errors.append(
                f"role_launch_meta['{role}'].launcher_only_aliases={list(extras)!r} has no "
                f"`parity.exceptions` entry. A launcher-only alias needs a written reason."
            )
        else:
            used_exceptions.add(key)

    # --- 4. the KV-feasibility group must be internally coherent -------------
    # `slots_by_shape` is the one declaration in master that is only meaningful
    # against the topology, so it is the one that can go stale silently: rename a
    # shape class, retire a role's halves, or mistype a key, and a role quietly
    # falls back to the flat scalar with no error anywhere. Four teeth:
    #   (a) every key is a real shape class,
    #   (b) every key matches at least one instance the role actually has,
    #   (c) every instance of the role resolves to a positive slot count,
    #   (d) the flat compat scalar equals the primary instance's resolved value.
    # (d) is the one that matters most in practice: `slots` is read directly by
    # consumers that know nothing about shapes, so a group that disagrees with it
    # ships two different answers to the same question.
    for role in sorted(set(_MANIFEST["role_launch_meta"]) | set(NUMA_CONFIG)):
        table, source = slots_by_shape_for(role)
        instance_classes = NUMA_INSTANCE_SHAPE_CLASSES.get(role, ())
        if table:
            unknown = sorted(set(table) - set(CPU_SHAPE_CLASSES))
            if unknown:
                errors.append(
                    f"slots_by_shape for role '{role}' ({source}) names unknown shape "
                    f"class(es) {unknown}; known classes are {sorted(CPU_SHAPE_CLASSES)}"
                )
            unused = sorted(set(table) - set(instance_classes) - set(unknown))
            if unused:
                errors.append(
                    f"slots_by_shape for role '{role}' ({source}) declares {unused} but the "
                    f"role has no instance of that class in stack_topology.yaml "
                    f"(its instances are {list(instance_classes)}). A slot count nothing "
                    f"can consume is a declaration that will never be noticed as wrong."
                )
        if not instance_classes:
            continue
        for idx, shape_class in enumerate(instance_classes):
            decision = resolve_slots(role, numa_instance=idx)
            if decision.slots <= 0:
                errors.append(
                    f"role '{role}' instance {idx} ({shape_class}) resolves to "
                    f"{decision.slots} slots via {decision.source}"
                )
        flat, flat_source = master_declared(role, "slots")
        primary_idx = NUMA_CONFIG.get(role, {}).get("full_instance_idx", 0) or 0
        primary = resolve_slots(role, numa_instance=primary_idx)
        if table and _positive_int(flat) is not None and flat != primary.slots:
            errors.append(
                f"role '{role}': flat `slots` ({flat_source}) is {flat!r} but the primary "
                f"instance (index {primary_idx}, class "
                f"{instance_classes[primary_idx] if primary_idx < len(instance_classes) else '?'}) "
                f"resolves to {primary.slots} via {primary.source}. The flat key is the COMPAT "
                f"SCALAR for consumers that do not know about shapes; it must agree with the "
                f"shape the endpoint actually serves."
            )

    stale = sorted(exceptions - used_exceptions)
    for field, role in stale:
        errors.append(
            f"parity exception {field}/{role} no longer applies — the two sides agree. "
            f"Delete it so the next real divergence is not pre-excused."
        )

    if errors:
        raise ValueError(
            "Launcher/master declaration parity violated (launch_manifest.yaml vs "
            "model_registry.yaml):\n  " + "\n  ".join(errors)
        )


# =============================================================================
# Capacity guard — the KV-feasibility group, validated TOGETHER
# =============================================================================
# `validate_declaration_parity()` checks that nobody declared the same thing
# twice. This checks something no parity rule can: that the numbers, each
# individually plausible, ADD UP on the hardware.
#
# The failure it exists to prevent has no single wrong line. Raise n_ctx here,
# add an instance there, switch a KV type back to f16 somewhere else — three
# defensible commits, and the fourth one OOMs a card at model-load time with a
# HIP allocation failure that names none of them. Grouping the three fields into
# `serving_shape` is what makes the sum computable from ONE declaration; this
# function is why the grouping was worth doing.
#
# ── THE ARITHMETIC ───────────────────────────────────────────────────────────
#     KV KiB/token @f16 = KV_LAYERS * head_count_kv * (key_length + value_length) * 2 / 1024
# read out of the GGUF header, and declared per role as
# `serving_shape.kv_kib_per_token_f16`.
#
# KV_LAYERS IS NOT ALWAYS block_count. This formula used to say block_count, and that
# is wrong for every model whose attention is not uniformly global. When the GGUF
# declares `<arch>.full_attention_interval = N`, only every Nth layer keeps a KV cache
# and the rest are filtered out of it entirely; the server says so itself, e.g.
#     llama_kv_cache: size = 2176.00 MiB ( 65536 cells,  16 layers, ...)
# for Qwen3.8-27B, which has block_count 65 and full_attention_interval 4 -> 16 layers.
#
#     KV_LAYERS = block_count // full_attention_interval   (when the key is present)
#     KV_LAYERS = block_count                              (otherwise)
#
# Measured 2026-09-22 against the live v10 GPU build, from the server's own KV buffer
# report at n_ctx 65536: f16 4096 MiB, q8_0 2176 MiB, q4_0 1152 MiB -> 64.0 / 34.0 /
# 18.0 KiB per token. The block_count form gives 260.0 KiB/token f16 for the same
# model: 4.06x too high.
#
# The error direction matters. Over-counting KV makes this gate REFUSE lineups that
# fit, and it did: a 262144-context role was scored as infeasible on a card with
# 16 GiB to spare, which nearly bought a KV-quantisation quality tradeoff that was
# never needed. Six fleet models are affected -- every Qwen3.6/3.8 -- because they all
# carry full_attention_interval 4. Models without the key (Qwen3-Next-80B, gemma-4)
# are declared correctly. K and V are quantised INDEPENDENTLY, so
# each side is scaled separately: f16 = 1.0, q8_0 = 0.5, q4_0 = 0.25 of that
# side's half. Total KV bytes for an instance = KiB/token * n_ctx; `-np` does
# NOT multiply it (llama-server partitions one -c-sized cache across slots).

# Byte-width of one KV element RELATIVE TO f16, per llama.cpp cache type.
# DERIVED FROM ggml's BLOCK LAYOUT, not from the nominal bit-width. A quantised
# KV block stores its scale(s) alongside the quants, so q8_0 is 34 bytes per 32
# elements (1.0625 B/elem), NOT 1.0 — the naive "half of f16" understates it by
# 5.88%, and q4_0 by 11.11%.
#
# This was wrong here until 2026-09-22, and the shape of the error is what gives
# it away: q5_0 and q5_1 already carried their true ratios (0.34375, 0.375) while
# q8_0, q4_0 and q4_1 carried nominal fractions (0.5, 0.25, 0.28125). A deliberate
# simplification would have been uniformly nominal; a half-corrected table is an
# oversight. The gate therefore UNDERSTATED KV for the three most-used types and
# could pass an infeasible lineup — precisely the failure the comment below says
# must never happen. Measured impact at the time: Qwen3.8-27B at n_ctx 262144,
# q8_0/q8_0, was scored 32.50 GiB against a true 34.53 GiB.
#
# Block sizes are asserted by ggml itself in ggml/src/ggml-common.h:
#   block_q4_0 = half + 32/2      = 18 B      block_q4_1 = 2*half + 32/2       = 20 B
#   block_q5_0 = half + u32 + 32/2 = 22 B     block_q5_1 = 2*half + u32 + 32/2 = 24 B
#   block_q8_0 = half + 32         = 34 B
# ratio = (block_bytes / 32) / 2.0, i.e. bytes-per-element relative to f16.
#
# Only the types this fleet actually uses are listed: an unrecognised type RAISES
# rather than being assumed f16, because assuming the LARGEST type would silently
# pass an infeasible lineup and assuming the smallest would silently fail a
# feasible one. Neither is a safe default, so there is none.
_GGML_QK = 32
_GGML_HALF = 2
_GGML_U32 = 4


def _block_ratio(block_bytes: int) -> float:
    """Element width relative to f16, from a ggml block's true size."""
    return (block_bytes / _GGML_QK) / 2.0


_KV_TYPE_F16_RATIO: dict[str, float] = {
    "f32": 2.0,
    "f16": 1.0,
    "bf16": 1.0,
    "q8_0": _block_ratio(_GGML_HALF + _GGML_QK),                        # 34 B -> 0.53125
    "q5_1": _block_ratio(2 * _GGML_HALF + _GGML_U32 + _GGML_QK // 2),   # 24 B -> 0.375
    "q5_0": _block_ratio(_GGML_HALF + _GGML_U32 + _GGML_QK // 2),       # 22 B -> 0.34375
    "q4_1": _block_ratio(2 * _GGML_HALF + _GGML_QK // 2),               # 20 B -> 0.3125
    "q4_0": _block_ratio(_GGML_HALF + _GGML_QK // 2),                   # 18 B -> 0.28125
}

_GIB_PER_KIB_TOKEN = 1.0 / (1024.0 * 1024.0)


def kv_layers(block_count: int, full_attention_interval: int | None) -> int:
    """How many layers actually keep a KV cache.

    NOT block_count when the GGUF declares `<arch>.full_attention_interval`: only
    every Nth layer is globally attentive and the others are filtered out of the
    cache. See the arithmetic note above for the measurement this comes from.
    """
    if not full_attention_interval or full_attention_interval <= 1:
        return int(block_count)
    return int(block_count) // int(full_attention_interval)


def kv_kib_per_token_f16(block_count: int, head_count_kv: int, key_length: int,
                         value_length: int, full_attention_interval: int | None = None) -> float:
    """The value a role's `serving_shape.kv_kib_per_token_f16` must declare.

    Every input is a GGUF header fact. Compute the declaration with this rather than
    by hand: hand-computing it with block_count is what put a 4.06x over-count into
    six fleet rows.
    """
    layers = kv_layers(block_count, full_attention_interval)
    return layers * int(head_count_kv) * (int(key_length) + int(value_length)) * 2 / 1024.0


def kv_gib_for(role: str, n_ctx: int, kv_quant: tuple[str, str] | None) -> float:
    """KV cache size in GiB for one instance of `role` at `n_ctx`."""
    shape, source = master_serving_shape(role)
    per_token_f16 = shape.get("kv_kib_per_token_f16")
    if not isinstance(per_token_f16, (int, float)) or per_token_f16 <= 0:
        raise ValueError(
            f"stack_manifest: role '{role}' has a serving_shape ({source}) with no usable "
            f"`kv_kib_per_token_f16`. It is a MEASURED GGUF fact "
            f"(block_count * head_count_kv * (key_length + value_length) * 2 / 1024) and the "
            f"capacity check cannot be run without it. Read it from the GGUF header rather "
            f"than estimating."
        )
    half = float(per_token_f16) / 2.0  # the K side and the V side, each
    k_type, v_type = (kv_quant or ("f16", "f16"))
    ratios = []
    for side, kv_type in (("k", k_type), ("v", v_type)):
        ratio = _KV_TYPE_F16_RATIO.get(str(kv_type).lower())
        if ratio is None:
            raise ValueError(
                f"stack_manifest: role '{role}' declares KV type {kv_type!r} for the {side} "
                f"side, which the capacity check has no width for. Add it to "
                f"_KV_TYPE_F16_RATIO with its true element width — do NOT let it default."
            )
        ratios.append(ratio)
    return half * (ratios[0] + ratios[1]) * float(n_ctx) * _GIB_PER_KIB_TOKEN


def serving_shape_instances() -> list[dict]:
    """One record per declared instance, with its resolved shape and footprint.

    The JOIN made explicit: model facts from the master registry, placement from
    stack_topology.yaml, one row each. Roles with no `serving_shape` block are
    skipped — that is every launcher-only lane (`eval_batch_frontdoor`), the
    embedder pool and `worker_fast`, none of which master describes.
    """
    rows: list[dict] = []
    for role in sorted(set(_MANIFEST["role_launch_meta"]) & set(NUMA_CONFIG)):
        shape, shape_source = master_serving_shape(role)
        if not shape:
            continue
        n_ctx = shape.get("n_ctx")
        if not isinstance(n_ctx, int) or isinstance(n_ctx, bool) or n_ctx <= 0:
            raise ValueError(
                f"stack_manifest: role '{role}' has a serving_shape ({shape_source}) with no "
                f"usable `n_ctx`. The group is validated as a whole; a partial group cannot be."
            )
        kv_quant = LAUNCH_KV_QUANT_CONFIGS.get(role)
        device, _device_source = master_declared(role, "device")
        classes = NUMA_INSTANCE_SHAPE_CLASSES.get(role, ())
        for idx, instance in enumerate(NUMA_CONFIG[role]["instances"]):
            shape_class = classes[idx] if idx < len(classes) else None
            rows.append(
                {
                    "role": role,
                    "numa_instance": idx,
                    "port": int(instance[1]),
                    "shape_class": shape_class,
                    "device": str(device) if isinstance(device, str) else None,
                    "on_gpu": shape_class == "gpu_host_lane",
                    "n_ctx": n_ctx,
                    "slots": resolve_slots(role, numa_instance=idx).slots,
                    "kv_quant": kv_quant,
                    "kv_gib": round(kv_gib_for(role, n_ctx, kv_quant), 4),
                    "vram_non_kv_gib": shape.get("vram_non_kv_gib"),
                    "host_weights_gib": master_declared(role, "memory_gb")[0],
                    "no_mmap": bool(master_declared(role, "no_mmap")[0]),
                }
            )
    return rows


def _host_memtotal_gib() -> float | None:
    """Host RAM from /proc/meminfo. A HOST FACT, read-only, never a knob.

    Returns None when unreadable (a container without /proc, a non-Linux test
    box). The host gate is then reported as UNGATED rather than passing quietly —
    "capacity unknown" and "capacity fine" must not look the same.
    """
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        return None
    return None


def serving_shape_capacity_report() -> dict:
    """Per-device footprint of the declared lineup. Pure arithmetic, no processes.

    GPU is gated through `src.scheduling.device_model.vram_fit` rather than a
    second capacity model: that module already owns "does the resident set fit",
    including the declared 63.98 GiB total, the 2.0 GiB first-execution headroom
    and the alias-collapse rule. It is handed a SYNTHETIC `server_mode` in which
    each GPU role's VRAM is `vram_non_kv_gib + kv_gib` for its declared shape,
    with `priors={}` so it cannot reach past the closed world it was given.

    Host has no such helper — device_model is VRAM-only — so the host sum lives
    here, sharing the same per-instance KV numbers so the two can never disagree
    about what a KV cache costs.
    """
    from src.scheduling.device_model import vram_fit

    rows = serving_shape_instances()

    gpu_rows = [row for row in rows if row["on_gpu"]]
    synthetic_server_mode: dict[str, dict] = {}
    missing_vram: list[str] = []
    for row in gpu_rows:
        non_kv = row["vram_non_kv_gib"]
        if not isinstance(non_kv, (int, float)) or non_kv <= 0:
            missing_vram.append(row["role"])
            continue
        # One entry per SERVER (primary role only — aliases are never in
        # role_launch_meta), so vram_fit's alias collapse has nothing to collapse
        # and cannot double-count a shared process.
        synthetic_server_mode[row["role"]] = {
            "vram_gib": round(float(non_kv) + row["kv_gib"], 4)
        }
    if missing_vram:
        raise ValueError(
            "stack_manifest: GPU role(s) "
            f"{sorted(set(missing_vram))} declare no `serving_shape.vram_non_kv_gib`. "
            "The capacity check needs the KV-FREE resident figure; `vram_gib` is a "
            "KV-inclusive total at some past context and using it would double-count."
        )
    fit = vram_fit(
        [row["role"] for row in gpu_rows],
        priors={},
        server_mode=synthetic_server_mode,
    )

    host_rows = [row for row in rows if not row["on_gpu"]]
    host_kv_gib = round(sum(row["kv_gib"] for row in host_rows), 4)
    # Weights: a `no_mmap` role gets a PRIVATE copy per instance (that is the
    # point of the 2026-07-30 change — shared mmap placed every page once, on
    # whichever node faulted first). A shared-mmap role's GGUF is resident once.
    host_weights_gib = 0.0
    counted_shared: set[str] = set()
    for row in host_rows:
        weights = row["host_weights_gib"]
        if not isinstance(weights, (int, float)) or weights <= 0:
            continue
        if row["no_mmap"]:
            host_weights_gib += float(weights)
        elif row["role"] not in counted_shared:
            host_weights_gib += float(weights)
            counted_shared.add(row["role"])
    host_weights_gib = round(host_weights_gib, 4)

    host_total = _host_memtotal_gib()
    reserve = float(_CAPACITY["host_os_reserve_gib"])
    host_required = round(host_kv_gib + host_weights_gib, 4)
    host_budget = round(host_total - reserve, 4) if host_total is not None else None
    return {
        "instances": rows,
        "gpu": {
            "fit": fit,
            "required_gib": fit.required_gib,
            "budget_gib": fit.budget_gib,
            "capacity_gib": fit.capacity_gib,
            "per_role": dict(fit.per_role),
            "kv_gib": round(sum(row["kv_gib"] for row in gpu_rows), 4),
        },
        "host": {
            "gated": host_budget is not None,
            "kv_gib": host_kv_gib,
            "weights_gib": host_weights_gib,
            "required_gib": host_required,
            "budget_gib": host_budget,
            "capacity_gib": round(host_total, 4) if host_total is not None else None,
            "reserve_gib": reserve,
            "ok": True if host_budget is None else host_required <= host_budget,
        },
    }


def validate_serving_shape_capacity() -> None:
    """Refuse a lineup whose declared KV does not fit the hardware it runs on.

    Fails CLOSED on the GPU, where the margin is 3 GiB and the failure mode is a
    load-time HIP allocation error. Fails closed on the host too when
    /proc/meminfo is readable; when it is not, the host leg is reported UNGATED
    and only the GPU leg binds — a container that cannot see host RAM must not
    be able to pass a check it never ran.

    KNOWN UNDERCOUNT, recorded rather than papered over: whisper.cpp (STT) and
    Qwen3-TTS also tenant this MI210 when active. Neither has a `server_mode` row
    — voice_server still describes a CPU faster-whisper service and TTS has no
    entry at all — so the GPU sum is ~2.5 GiB optimistic whenever they are up.
    Both were ABSENT from the card at the 2026-08-02 measurement (rocm-smi
    --showpids showed only the two llama-servers). The 2.0 GiB `vram_fit`
    headroom does not cover them; giving them registry rows is the real fix.
    """
    report = serving_shape_capacity_report()
    problems: list[str] = []

    fit = report["gpu"]["fit"]
    if not fit.ok:
        overage = round(fit.required_gib - fit.budget_gib, 4)
        detail = ", ".join(
            f"{role} {gib:.2f}" for role, gib in sorted(fit.per_role.items())
        )
        problems.append(
            f"device ROCm0 (GPU) OVERSUBSCRIBED by {overage:.2f} GiB: declared lineup needs "
            f"{fit.required_gib:.2f} GiB but the budget is {fit.budget_gib:.2f} GiB "
            f"({fit.capacity_gib:.2f} GiB capacity from {fit.capacity_source} minus "
            f"{fit.headroom_gib:.2f} GiB headroom). Per role (weights+KV): {detail}. "
            f"Reason: {fit.reason or 'n/a'}. Lower `serving_shape.n_ctx`, or quantise "
            f"`serving_shape.kv_quant` further, for a role on this device."
        )

    host = report["host"]
    if host["gated"] and not host["ok"]:
        overage = round(host["required_gib"] - host["budget_gib"], 4)
        problems.append(
            f"device host (CPU RAM) OVERSUBSCRIBED by {overage:.2f} GiB: declared lineup needs "
            f"{host['required_gib']:.2f} GiB ({host['weights_gib']:.2f} weights + "
            f"{host['kv_gib']:.2f} KV) but the budget is {host['budget_gib']:.2f} GiB "
            f"({host['capacity_gib']:.2f} GiB MemTotal minus {host['reserve_gib']:.2f} GiB "
            f"OS reserve)."
        )

    if problems:
        raise ValueError(
            "Declared serving_shape lineup does not fit the hardware "
            "(model_registry.yaml server_mode.<role>.serving_shape x "
            "stack_topology.yaml placement):\n  " + "\n  ".join(problems)
        )


# Validate at module load (fails fast on misconfiguration)
_validate_role_classification()
validate_declaration_parity()
validate_serving_shape_capacity()

# Computed server lists. Single source of truth: ROLE_LAUNCH_META + NUMA_CONFIG.
HOT_SERVERS, WARM_SERVERS = _build_servers_from_classification()


def validate_against_registry(
    registry_yaml_path: str | None = None,
    *,
    roles: set[str] | None = None,
) -> list[str]:
    """Cross-check ROLE_LAUNCH_META against orchestration/model_registry.yaml.

    Returns list of warning strings (empty if everything is consistent).
    Called from `start` command but non-fatal: prints warnings, does not abort.
    Useful for catching drift between launcher classification and registry's
    process_layout / server_mode sections.
    """
    if registry_yaml_path is None:
        # Default: orchestration/model_registry.yaml relative to repo root
        registry_yaml_path = str(
            Path(__file__).parent.parent.parent / "orchestration" / "model_registry.yaml"
        )
    warnings: list[str] = []
    role_scope = set(roles) if roles is not None else None

    def in_scope(role: str) -> bool:
        return role_scope is None or role in role_scope

    try:
        import yaml

        with open(registry_yaml_path) as f:
            registry = yaml.safe_load(f)
    except Exception as e:
        return [f"could not load registry from {registry_yaml_path}: {e}"]

    pl = registry.get("process_layout", {})
    has_process_layout = isinstance(pl, dict) and any(
        key in pl for key in ("hot_resident", "warm_mmap")
    )
    reg_hot = set(pl.get("hot_resident", [])) if isinstance(pl, dict) else set()
    reg_warm = set(pl.get("warm_mmap", [])) if isinstance(pl, dict) else set()

    # Compute launcher's HOT tier including aliases (shared_with_first_n).
    # Aliases share the same process as their primary, so for tier comparison
    # they count as HOT too.
    launcher_hot: set[str] = set()
    launcher_warm: set[str] = set()
    for role, meta in ROLE_LAUNCH_META.items():
        target = launcher_hot if meta["tier"] == "hot" else launcher_warm
        target.add(role)
        for alias in meta.get("shared_with_first_n", []):
            target.add(alias)

    # Production roles to cross-check — skip launcher-only aliases not present
    # in the registry process_layout. This keeps the validator aligned with the
    # live hot/warm sets instead of a hand-maintained internal-name list.
    skip = {role for role in launcher_hot if role not in reg_hot and role not in reg_warm}
    launcher_hot_filtered = {role for role in launcher_hot - skip if in_scope(role)}
    reg_hot_filtered = {role for role in reg_hot if in_scope(role)}
    reg_warm_filtered = {role for role in reg_warm if in_scope(role)}

    if has_process_layout:
        only_in_launcher = launcher_hot_filtered - reg_hot_filtered - reg_warm_filtered
        only_in_registry_hot = reg_hot_filtered - launcher_hot - launcher_warm

        for r in sorted(only_in_launcher):
            warnings.append(
                f"role '{r}' is HOT in launcher but absent from registry process_layout"
            )
        for r in sorted(only_in_registry_hot):
            # Roles that the registry says should be hot but launcher doesn't classify hot
            warnings.append(
                f"role '{r}' is hot_resident in registry but not in launcher's HOT tier"
            )

    # Cross-check legacy/direct role port hints against computed launch roles.
    # PORT_MAP is still used by targeted reload/status compatibility paths; it
    # must agree with ROLE_LAUNCH_META + NUMA_CONFIG for shared aliases.
    computed_role_ports: dict[str, int] = {}
    for server in HOT_SERVERS + WARM_SERVERS:
        port = server.get("port")
        if not isinstance(port, int):
            continue
        for role in server.get("roles", []):
            computed_role_ports.setdefault(str(role), port)
    for role, port in PORT_MAP.items():
        if not in_scope(role):
            continue
        computed_port = computed_role_ports.get(role)
        if computed_port is not None and port != computed_port:
            warnings.append(
                f"role '{role}': PORT_MAP says port {port}, "
                f"but computed launch roles use port {computed_port}"
            )

    # Cross-check NUMA_CONFIG ports vs registry server_mode ports (where present)
    sm = registry.get("server_mode", {})
    for role, srv in sm.items():
        if not isinstance(srv, dict):
            continue
        covered_roles = {str(role)}
        model_role = srv.get("model_role")
        if isinstance(model_role, str):
            covered_roles.add(model_role)
        shared_with = srv.get("shared_with")
        if isinstance(shared_with, list):
            covered_roles.update(str(item) for item in shared_with if isinstance(item, str))
        if role_scope is not None and not covered_roles & role_scope:
            continue
        reg_port = srv.get("port")
        if reg_port is None:
            continue
        cfg = NUMA_CONFIG.get(role)
        if cfg:
            launcher_ports = [inst[1] for inst in cfg["instances"]]
            if reg_port not in launcher_ports:
                warnings.append(
                    f"role '{role}': registry server_mode says port {reg_port}, "
                    f"but launcher NUMA_CONFIG ports are {launcher_ports}"
                )
        else:
            meta = ROLE_LAUNCH_META.get(role)
            if meta and meta.get("no_numa") and meta.get("port") != reg_port:
                warnings.append(
                    f"role '{role}': registry server_mode says port {reg_port}, "
                    f"but launcher ROLE_LAUNCH_META says port {meta.get('port')}"
                )
            if not meta:
                for covered_role in sorted(covered_roles):
                    if not in_scope(covered_role):
                        continue
                    computed_port = computed_role_ports.get(covered_role)
                    if computed_port is not None and reg_port != computed_port:
                        warnings.append(
                            f"role '{role}': registry server_mode says port {reg_port} "
                            f"for launch role '{covered_role}', "
                            f"but computed launch roles use port {computed_port}"
                        )
    return warnings


# =============================================================================
# Model Path Validation
# =============================================================================


def validate_model_paths() -> list[str]:
    """Validate all model paths exist. Returns list of errors.

    This prevents hallucinations about missing models by failing fast
    with clear error messages showing exactly what's missing.
    """
    errors = []

    # HOT tier models
    if not Path(EMBEDDING_MODEL_PATH).exists():
        errors.append(f"[HOT] Embedding: {EMBEDDING_MODEL_PATH}")

    for worker_type, path in WORKER_POOL_MODELS.items():
        if not Path(path).exists():
            errors.append(f"[HOT] Worker '{worker_type}': {path}")

    # Draft model for explore worker spec decode. `models.explore_draft_model` is
    # NULLABLE by design (launch_manifest.yaml): a NEXTN/MTP lane FUSES its draft head
    # into the base GGUF, so there is no separate drafter file to validate. `None` means
    # "no external drafter", not "a missing file". Unguarded this raised
    # `TypeError: expected str, bytes or os.PathLike object, not NoneType` inside the
    # `[0.5] Validating model paths` preflight (stack_commands.py), i.e. a non-dev stack
    # start crashed before launching anything. Every other consumer of the constant
    # already treats it as nullable (stack_change_guard.py, stack_priors.py,
    # orchestrator_stack.py); this was the one site that did not.
    if EXPLORE_DRAFT_MODEL and not Path(EXPLORE_DRAFT_MODEL).exists():
        errors.append(f"[HOT] Explore draft: {EXPLORE_DRAFT_MODEL}")

    # Serving artifacts for the fleet-hosting roles, DERIVED from the row master says
    # governs each one -- the same binding the compiled priors use.
    #
    # 2026-09-23 (SSU-F8). All three were hardcoded literals and all three were wrong:
    #   frontdoor            named .../Qwen_Qwen3.6-35B-A3B-Q8_0.gguf, NOT the
    #                        ...-MTP-Q8_0.gguf it actually serves. The file it named
    #                        happened to exist, so the preflight reported green while
    #                        validating the wrong artifact.
    #   architect_general    named .../unsloth/Qwen3.5-122B-A10B-GGUF/
    #   ingest_long_context  named .../lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/
    #                        -- both retired directories that no longer exist on disk, so
    #                        once the EXPLORE_DRAFT_MODEL TypeError above stopped masking
    #                        this function, the `[0.5] Validating model paths` preflight
    #                        would have failed a non-dev stack start on two phantom paths.
    # A role whose master row declares no path is not this check's business; the
    # declaration-parity guard owns that.
    # 2026-05-06: architect_coding REMOVED (REAP-246B role eliminated; 139 GB freed).
    # 2026-09-27 ARCHITECT SWAP: architect_critic added — it is now the host row
    # of the :8083 27B GGUF; architect_general is the :8074 Flash-Next row.
    for role in ("frontdoor", "architect_general", "architect_critic", "ingest_long_context"):
        declared = _master_declared_model_path(role)
        if declared and not Path(declared).exists():
            errors.append(f"[HOT] {role}: {declared}")

    # Vision models (VL with multimodal projector)
    for label, path in [
        ("worker_vision model", VISION_WORKER_MODEL),
        ("worker_vision mmproj", VISION_WORKER_MMPROJ),
        ("vision_escalation model", VISION_ESCALATION_MODEL),
        ("vision_escalation mmproj", VISION_ESCALATION_MMPROJ),
    ]:
        if not Path(path).exists():
            errors.append(f"[HOT] {label}: {path}")

    # Auxiliary services
    formalizer = _PATHS["models_dir"] / "LightOnOCR-2-1B-bbox-Q4_K_M.gguf"
    if not formalizer.exists():
        errors.append(f"[AUX] document_formalizer: {formalizer}")

    # Tool registry (required for deterministic tools)
    tool_registry = _PATHS["project_root"] / "orchestration/tool_registry.yaml"
    if not tool_registry.exists():
        errors.append(f"[TOOL] tool_registry.yaml: {tool_registry}")

    # C++ math tools (optional - warn but don't fail)
    cpp_math_tools = LLAMA_MATH_TOOLS
    if not cpp_math_tools.exists():
        # This is a warning, not an error - append with different prefix
        pass  # Will be checked separately in init_memrl_and_tools

    return errors
