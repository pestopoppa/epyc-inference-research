"""Dashboard topology helpers — port → role discovery, color resolution, process info.

Pure-data helpers extracted from src/api/routes/dashboard.py during the 2026-05-21
refactor. Route handlers in dashboard.py re-import these so signatures stay
unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from scripts.server.realized_fleet import derive_realized_numa_mode
from scripts.server.runtime_facts_manifest import (
    read_runtime_stack_numa_mode,
    read_runtime_stack_selected_servers,
    runtime_facts_manifest_path,
)
from scripts.server.stack_numa_mode import (
    DASHBOARD_RUNTIME_FALLBACK_NUMA_MODE,
    VALID_STACK_NUMA_MODES,
)
from src.roles import Role
from src.registry.stack_priors import (
    live_stack_role_records,
    stack_prior_serving,
    stack_prior_serving_ports,
)

logger = logging.getLogger(__name__)

_DEFAULT_STACK_PRIORS_PATH = (
    Path(__file__).resolve().parents[3] / "orchestration" / "derived" / "stack_priors.yaml"
)

# Service-only hints. Model-serving ports are projected from generated stack
# priors below so dashboard labels follow the same launch contract as the stack.
_BASE_SERVICE_PORT_HINTS: dict[int, str] = {
    8000: "orchestrator",
    8088: "nextplaid-code",
    8089: "nextplaid-docs",
    8090: "embedder",
    8091: "embedder_1",
    8092: "embedder_2",
    8093: "embedder_3",
    8094: "embedder_4",
    8095: "embedder_5",
}


def _aux_service_port_hints() -> dict[int, str]:
    """Auxiliary services, DERIVED from the launch manifest.

    These used to be restated here as literals (`8190: sd_server`,
    `9000: whisper`, `9001: document_formalizer`). The list drifted the moment a
    service was added: TTS went live on :9002 on 2026-08-02 and the handoff
    dashboard sits on :8100, and neither appeared here — so the dashboard showed
    a fleet the stack does not have. `AUX_SERVICES` is the same table `start`
    and `reload` dispatch off, so deriving from it means a newly declared
    service is visible without a second edit here.

    Lazy import: this module is imported by API routes, and stack_manifest pulls
    in the launcher surface. Failure degrades to {} — the model-serving ports,
    which are projected from stack priors below, are unaffected.
    """
    try:
        from scripts.server.stack_manifest import AUX_SERVICES

        return {
            int(svc.port): str(svc.name)
            for svc in AUX_SERVICES.values()
            if isinstance(getattr(svc, "port", None), int)
        }
    except Exception:  # noqa: BLE001
        return {}


def _service_port_hints() -> dict[int, str]:
    # Declared aux services win over the static hints: the manifest is the
    # launch contract, this dict is only for ports no manifest owns.
    hints = dict(_BASE_SERVICE_PORT_HINTS)
    hints.update(_aux_service_port_hints())
    return hints


def _label_for_stack_prior_entry(role: str, entry: dict[str, Any]) -> tuple[int, str] | None:
    if entry.get("alias"):
        return None
    primary_role = entry.get("primary_role")
    if isinstance(primary_role, str) and primary_role and primary_role != role:
        return None
    port = entry.get("port")
    if not isinstance(port, int):
        return None
    numa_instance = entry.get("numa_instance")
    if isinstance(numa_instance, int) and numa_instance > 0:
        return port, f"{role}.q{numa_instance - 1}"
    return port, role


def _stack_prior_port_hints(
    stack_priors_path: Path = _DEFAULT_STACK_PRIORS_PATH,
) -> dict[int, str]:
    """Project live model-serving port labels from generated stack priors."""
    roles = live_stack_role_records(stack_priors_path)
    if not roles:
        return {}

    hints: dict[int, str] = {}
    for role, record in sorted(roles.items()):
        serving = stack_prior_serving(record)
        launch = serving.get("launch")
        launch = launch if isinstance(launch, dict) else {}
        primary_roles = launch.get("primary_roles")
        if isinstance(primary_roles, list) and primary_roles and role not in primary_roles:
            continue

        mapped = False
        entries = launch.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                label = _label_for_stack_prior_entry(role, entry)
                if label is None:
                    continue
                port, name = label
                hints[port] = name
                mapped = True

        if mapped:
            continue
        ports = stack_prior_serving_ports(serving)
        for index, port in enumerate(ports):
            hints[port] = role if index == 0 else f"{role}.q{index - 1}"

    return hints


def _build_port_hints() -> dict[int, str]:
    hints = _service_port_hints()
    hints.update(_stack_prior_port_hints())
    return hints


# Public compatibility map used by tests and dashboard callers.
_PORT_HINTS: dict[int, str] = _build_port_hints()


# Realized-fleet probe cache. The dashboard must render the REALIZED live fleet,
# not launch-time intent (ESC-8 / audit finding C1): a uvicorn worker that
# inherited ORCHESTRATOR_STACK_NUMA_MODE=full must never paint a quarters-only
# fleet as rogue. The probe is a handful of bare-TCP connect_ex() calls against
# localhost (scripts.server.realized_fleet.derive_realized_numa_mode); a short
# TTL collapses a 2 Hz × 6-worker poll burst into a single probe so dashboard
# polling does not storm the loopback port universe.
_REALIZED_NUMA_CACHE: dict[str, Any] = {"ts": 0.0, "value": None, "probed": False}
_REALIZED_NUMA_TTL_S = 5.0


def _probe_realized_numa_mode() -> str | None:
    """Probe the live fleet and classify its realized NUMA mode (``full`` /
    ``quarter`` / ``both``), or ``None`` when nothing in the quarterable port
    universe is listening / the probe fails.

    Isolated seam: production opens localhost TCP connections here; tests
    neutralize or substitute this without touching sockets.
    """
    try:
        return derive_realized_numa_mode()
    except Exception:
        logger.debug("realized-fleet NUMA probe failed; ignoring", exc_info=True)
        return None


def _cached_realized_numa_mode() -> str | None:
    """TTL-cached ``_probe_realized_numa_mode()`` (see the cache note above)."""
    now = time.monotonic()
    cache = _REALIZED_NUMA_CACHE
    if cache.get("probed") and (now - cache["ts"]) < _REALIZED_NUMA_TTL_S:
        return cache["value"]
    value = _probe_realized_numa_mode()
    cache["ts"] = now
    cache["value"] = value
    cache["probed"] = True
    return value


def _env_declared_numa_mode() -> str | None:
    """The env-declared stack NUMA mode, or ``None`` when unset/blank/unrecognized.

    Unlike ``env_stack_numa_mode`` (which coerces anything to a default), this
    returns ``None`` for absent or non-canonical values so the resolver can treat
    the env purely as a spawn-time hint that carries intent only when it is a real
    mode string.
    """
    raw = os.environ.get("ORCHESTRATOR_STACK_NUMA_MODE")
    if raw is None:
        return None
    normalized = raw.strip().lower()
    return normalized if normalized in VALID_STACK_NUMA_MODES else None


def active_stack_numa_mode_resolution() -> dict[str, Any]:
    """Resolve the stack NUMA mode the dashboard should render, realized-fleet first.

    Precedence (audit finding C1 — the dashboard renders the REALIZED fleet, not
    launch-time intent)::

        realized live fleet  >  hardened runtime-facts manifest  >  env hint  >  default

    ``ORCHESTRATOR_STACK_NUMA_MODE`` is demoted to a LAST-resort spawn-time hint.
    A uvicorn worker that inherited ``full`` must not flag a quarters-only fleet as
    rogue (the proven C1 inversion) — the realized probe overrides it. When a
    lower-precedence source contradicts the resolved mode, the contradiction is
    recorded in ``disagreements`` so surfaces can badge it ("env disagrees: full")
    instead of silently trusting the lying source.

    Returns a provenance dict:
      * ``mode``          — the string the surfaces render.
      * ``source``        — ``realized_fleet`` / ``runtime_manifest`` / ``env`` / ``default``.
      * ``realized`` / ``manifest`` / ``env`` — each source's value (str | None).
      * ``disagreements`` — human-readable lower-precedence contradictions.
    """
    realized = _cached_realized_numa_mode()
    manifest = _fail_closed_runtime_stack_numa_mode()
    env = _env_declared_numa_mode()

    if realized is not None:
        mode, source = realized, "realized_fleet"
        lower: list[tuple[str, str | None]] = [("manifest", manifest), ("env", env)]
    elif manifest is not None:
        mode, source = manifest, "runtime_manifest"
        lower = [("env", env)]
    elif env is not None:
        mode, source = env, "env"
        lower = []
    else:
        mode, source = DASHBOARD_RUNTIME_FALLBACK_NUMA_MODE, "default"
        lower = []

    disagreements = [
        f"{label} disagrees: {value}"
        for label, value in lower
        if value is not None and value != mode
    ]
    return {
        "mode": mode,
        "source": source,
        "realized": realized,
        "manifest": manifest,
        "env": env,
        "disagreements": disagreements,
    }


def active_stack_numa_mode() -> str:
    """Return the stack NUMA mode string the dashboard surfaces should render.

    Thin wrapper over :func:`active_stack_numa_mode_resolution` for the many call
    sites that need only the mode string. See that function for the
    realized-fleet-first precedence and provenance semantics. This drives only the
    dashboard/health family; the config compiler and launcher own their own
    spawn-time planning.
    """
    return active_stack_numa_mode_resolution()["mode"]


def _fail_closed_runtime_stack_numa_mode() -> str | None:
    """Return the runtime-facts stack NUMA mode ONLY when the manifest passes the
    same fail-closed contract the URL reader (read_runtime_stack_selected_servers)
    enforces: a concrete expected mode string AND a non-empty selected-server
    lineup consistent with the declared ports.

    WP-14: the launcher can leave a phantom full-era lineup behind (the real
    current shape is stack_numa_mode=None, selected_ports=[], full-era
    selected_servers). read_runtime_stack_numa_mode() alone would either accept a
    stale mode or return None while the topology port hints still projected the
    phantom lineup, so the dashboard could render a NUMA mode that no live
    process backs. Mirror the URL reader's rejection here and fall back to the
    dashboard's historical env/NUMA_CONFIG default with one loud log line.
    """
    mode = read_runtime_stack_numa_mode()
    servers = read_runtime_stack_selected_servers()
    lineup_ok = isinstance(servers, list) and bool(servers)
    if isinstance(mode, str) and mode and lineup_ok:
        return mode.strip().lower()

    try:
        manifest_present = runtime_facts_manifest_path().exists()
    except Exception:
        manifest_present = False
    if manifest_present:
        logger.warning(
            "runtime-facts manifest rejected (fail-closed: stack_numa_mode=%r, "
            "selected lineup %s); falling back to dashboard NUMA_CONFIG default",
            mode,
            "present" if lineup_ok else "empty/inconsistent",
        )
    return None


def _manifest_server_label(server: dict[str, Any]) -> str:
    roles = server.get("roles") or []
    role = str(roles[0]) if isinstance(roles, list) and roles else ""
    if not role:
        return ""
    numa_instance = server.get("numa_instance")
    if isinstance(numa_instance, int) and numa_instance > 0:
        return f"{role}.q{numa_instance - 1}"
    return role


def _manifest_port_hints(numa_mode: str | None = None) -> dict[int, str]:
    if numa_mode is None and os.environ.get("ORCHESTRATOR_STACK_NUMA_MODE") is None:
        runtime_servers = read_runtime_stack_selected_servers()
        if runtime_servers is not None:
            hints: dict[int, str] = {}
            for server in runtime_servers:
                port = server.get("port")
                label = _manifest_server_label(server)
                if isinstance(port, int) and label:
                    hints[port] = label
            return hints

    try:
        from scripts.server.stack_manifest import HOT_SERVERS, WARM_SERVERS, _filter_by_numa_mode
    except Exception:
        return {}
    mode = numa_mode or active_stack_numa_mode()
    try:
        servers = _filter_by_numa_mode(HOT_SERVERS + WARM_SERVERS, mode)
    except Exception:
        servers = HOT_SERVERS + WARM_SERVERS
    hints: dict[int, str] = {}
    for server in servers:
        if not isinstance(server, dict):
            continue
        port = server.get("port")
        label = _manifest_server_label(server)
        if isinstance(port, int) and label:
            hints[port] = label
    return hints


def _configured_numa_port_hints() -> dict[int, str]:
    """Labels for every statically configured NUMA llama-server port.

    This is deliberately independent of the active launch mode. A quarter
    listener that is already running must be labeled as `role.qN` in topology
    and lock surfaces even if the current manifest mode says that quarter was
    not expected for this run.
    """
    try:
        from scripts.server.stack_numa import NUMA_CONFIG
    except Exception:
        return {}

    hints: dict[int, str] = {}
    for role, cfg in (NUMA_CONFIG or {}).items():
        if not isinstance(cfg, dict):
            continue
        instances = cfg.get("instances")
        if not isinstance(instances, list):
            continue
        full_idx = cfg.get("full_instance_idx")
        for idx, entry in enumerate(instances):
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                continue
            port = entry[1]
            if not isinstance(port, int):
                continue
            label = role
            if isinstance(full_idx, int) and idx != full_idx:
                label = f"{role}.q{idx - 1 if idx > full_idx else idx}"
            hints[port] = label
    return hints


def _port_hint(port: int) -> str:
    return (
        _PORT_HINTS.get(port)
        or _manifest_port_hints().get(port)
        or _configured_numa_port_hints().get(port)
        or f"port_{port}"
    )

# Per-role display colors (CSS hex).
_ROLE_COLORS: dict[str, str] = {
    "mi210_gpu": "#f97316",
    "frontdoor": "#3b82f6",
    "worker_general": "#10b981",
    "worker_math": "#10b981",
    "architect_general": "#a855f7",
    "ingest_long_context": "#f59e0b",
    "coder_escalation": "#ef4444",
    "worker_summarize": "#06b6d4",
    "worker_vision": "#ec4899",
    "vision_escalation": "#ec4899",
    "embedder": "#94a3b8",
    "orchestrator": "#475569",
}


def base_role(role: str) -> str:
    """Collapse an instance/quarter label to its canonical base role.

    Mirrors the grouping the dashboard front-end (`renderTopologyStrip`) applies
    so that every surface — topology rows, slot-dot aggregation, in-flight task
    grouping, and the recent-activity headline — keys off the same string:

        "frontdoor.q2"  -> "frontdoor"   (NUMA quarter)
        "embedder_3"    -> "embedder"    (numbered sibling)
        "architect_general" -> unchanged

    Only a trailing `_<digits>` is stripped, so multi-word roles like
    `architect_general` / `ingest_long_context` are left intact.
    """
    if not role:
        return ""
    base = role.split(".")[0]
    return re.sub(r"_\d+$", "", base)


def _role_color(role: str) -> str:
    """Resolve a role label to its display color, falling back to gray.

    Strips both `.qN` (NUMA quarter) and `_N` (numbered siblings like
    embedder_1) suffixes before lookup.
    """
    base = role.split(".")[0]
    # Strip trailing _<digits> if the prefix is a known role family.
    m = re.match(r"^(.+?)_\d+$", base)
    if m and m.group(1) in _ROLE_COLORS:
        base = m.group(1)
    canonical = Role.from_string(base)
    if canonical is not None:
        base = canonical.value
    return _ROLE_COLORS.get(base, "#64748b")


def role_aliases(role: str) -> list[str]:
    """Return the list of alias role names served by the same llama-server.

    Reads `shared_with_first_n` from stack_manifest.ROLE_LAUNCH_META. e.g.
    `frontdoor` returns `["coder_escalation", "worker_summarize"]`. Returns []
    when the role has no aliases or when the manifest cannot be imported
    (test contexts, scripts run outside the stack tree).
    """
    base = base_role(role)
    try:
        # Lazy import — keeps dashboard_topology importable without the scripts
        # tree on sys.path (e.g. unit tests).
        import sys
        scripts_dir = Path(__file__).resolve().parents[3] / "scripts" / "server"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from stack_manifest import ROLE_LAUNCH_META  # type: ignore
        meta = ROLE_LAUNCH_META.get(base, {})
        aliases = meta.get("shared_with_first_n") or []
        return list(aliases)
    except Exception:
        return []


# --- Extern/unmanaged attribution (dashboard task M3, 2026-07-23) ------------
#
# A llama-server listener may render under a production-lane label ONLY when
# the stack can vouch for it:
#   - a fresh per-port fleet marker (/mnt/raid0/llm/tmp/llama_<port>_started_at,
#     written by stack_commands BEFORE Popen), or
#   - membership in the runtime launch contract (runtime-facts
#     ``runtime_stack.selected_servers``, the realized launcher lineup).
# Anything else is an unmanaged process squatting on (or near) a configured
# lane port — bench harnesses, GPU warmups (the observed extern_18072 class) —
# and must render as ``extern_<port>`` instead of inheriting a role from a
# stale marker or a static lineup (the observed bug: 18072 read as
# "eval_batch_frontdoor" via the stale 18070 marker / static runtime-facts
# lineup). See handoffs/active/autopilot-dashboard-fidelity-audit-2026-07-22.md
# (M3) + stack-lineup-dossier-2026-07-23.md §4 item 9 / §6 contradiction 2.

# Markers are written milliseconds before Popen, so a marker that predates the
# live listener by more than this belongs to a PREVIOUS process on that port.
_MARKER_STALE_TOLERANCE_S = 300.0

ATTRIBUTION_FLEET_MARKER = "fleet-marker"
ATTRIBUTION_LAUNCH_CONTRACT = "launch-contract"
ATTRIBUTION_SERVICE_HINT = "service-hint"
ATTRIBUTION_UNMANAGED = "unmanaged"
ATTRIBUTION_UNVERIFIED = "unverified"


def _llama_fleet_markers() -> dict[int, dict[str, Any]]:
    """Per-port fleet-startup markers ({port: {started_at, source, roles}}).

    Empty on any failure — attribution then falls through to the launch
    contract and, absent both planes, fails open (no demotion).
    """
    try:
        from scripts.server.fleet_markers import discover_llama_markers

        return discover_llama_markers()
    except Exception:
        return {}


def _launch_contract_ports() -> set[int]:
    """Ports in the current runtime launch contract (realized selected servers)."""
    try:
        servers = read_runtime_stack_selected_servers()
    except Exception:
        return set()
    if not servers:
        return set()
    return {s["port"] for s in servers if isinstance(s.get("port"), int)}


def _classify_llama_attribution(
    port: int,
    proc_started_at: float | None,
    markers: dict[int, dict[str, Any]],
    contract_ports: set[int],
) -> dict[str, Any]:
    """Classify how a live llama-server listener is vouched for by the stack.

    Returns ``{"attribution": <str>, "marker_stale": <bool>}`` where attribution
    is one of the ``ATTRIBUTION_*`` constants. ``unverified`` means the whole
    attribution plane carried no signal (no markers, no contract — dev
    checkouts, hermetic tests), in which case callers must fail open and keep
    legacy labels rather than demote a healthy fleet.
    """
    marker = markers.get(port)
    marker_stale = False
    if isinstance(marker, dict):
        marker_started = marker.get("started_at")
        if not isinstance(marker_started, (int, float)) or not isinstance(
            proc_started_at, (int, float)
        ):
            # Unknown timing on either side: a marker cannot vouch for a
            # process it cannot be matched to (verifier finding 2 — without
            # this, an 18-day-stale marker vouches whenever ps parsing fails).
            marker_stale = True
        elif float(proc_started_at) - float(marker_started) > _MARKER_STALE_TOLERANCE_S:
            # Marker predates this listener by more than a launch gap: it was
            # written for a previous stack process on this port (e.g. the
            # 2026-07-05 llama_18070 marker) and must not vouch for this one.
            marker_stale = True
        elif float(marker_started) - float(proc_started_at) > _MARKER_STALE_TOLERANCE_S:
            # Symmetric case (verifier finding 1): a listener that predates
            # the marker by more than a launch gap cannot be the process the
            # marker was written for — orchestrator_stack writes the marker
            # BEFORE Popen, so a long-running squatter on a lane port must
            # not inherit the label when a fresh launch attempt dies.
            marker_stale = True
        else:
            return {"attribution": ATTRIBUTION_FLEET_MARKER, "marker_stale": False}
    if port in contract_ports:
        return {"attribution": ATTRIBUTION_LAUNCH_CONTRACT, "marker_stale": marker_stale}
    if not markers and not contract_ports:
        return {"attribution": ATTRIBUTION_UNVERIFIED, "marker_stale": marker_stale}
    return {"attribution": ATTRIBUTION_UNMANAGED, "marker_stale": marker_stale}


def _ps_llama_scan() -> str:
    """Raw ``ps`` output for the llama process scan (patchable in unit tests)."""
    try:
        return subprocess.run(
            ["ps", "-eo", "pid,etimes,cmd"], capture_output=True, text=True, timeout=2,
        ).stdout
    except Exception:
        return ""


# Substrate from the BINARY path — never the model path, which is substrate-free.
# On this host the GPU inference kernels live in HIP/ROCm build trees
# (`.../llama.cpp/build-hip/bin/llama-server`), so the marker is in argv[0].
# Derived per PROCESS because role names carry no substrate (architect_general
# was a GPU role until the 2026-09-27 ARCHITECT SWAP made it the CPU :8074
# process, and nothing in its name said either), and a hardcoded
# role→substrate list is exactly the drift class RTG-47 removes.
_SUBSTRATE_MARKER_RE = re.compile(r"hip|rocm|gfx", re.IGNORECASE)


def _is_gpu_binary(path: str) -> bool:
    """True when `path` is served by an accelerator backend.

    Resolve through the KERNEL STORE first; fall back to the substring marker.

    Why: the marker above keys on the frozen source tree's directory NAMING
    (`llama.cpp/build-hip/bin`). When kernels/production/gpu was repointed at a
    versioned store dir on 2026-09-21 (`kernels/builds/gpu-<date>-<sha>/bin`) the
    substring vanished -- that path contains no "hip", "rocm" or "gfx" -- and every
    GPU role silently began reporting substrate "cpu". Nothing errors; the topology
    view just becomes quietly wrong, which is the same class of silent-wrongness the
    marker was introduced to avoid. Verified 2026-09-22: the regex matches the v9
    path and neither v10 path.

    The store mapping is authoritative because it is what the launcher resolves
    (src/registry/kernel_paths.backend_dir). Non-llama backends (stt/tts) are
    accelerator-backed on this host too, so anything that is not the `cpu` backend
    counts as GPU here, matching the previous marker's intent.
    """
    if not path:
        return False
    try:
        from pathlib import Path as _P

        from src.registry.kernel_paths import BACKEND_BINARIES, backend_dir

        resolved = _P(path).resolve()
        for backend in BACKEND_BINARIES:
            try:
                bdir = _P(backend_dir(backend)).resolve()
            except Exception:
                continue
            if resolved == bdir or bdir in resolved.parents:
                return backend != "cpu"
    except Exception:
        pass
    return bool(_SUBSTRATE_MARKER_RE.search(path))

# Manifest `device` strings that DECLARE a GPU lane (model_registry.yaml
# server_mode.<role>.serving_shape.device, e.g. "ROCm0"). Host lanes declare no
# device at all — the manifest's own convention (`serving_shape_capacity_report`
# books a device-less role against host RAM), so omission reads as cpu HERE and
# only here, where the manifest is the declared source of truth.
_MANIFEST_GPU_DEVICE_RE = re.compile(r"rocm|hip|gfx|gpu|cuda", re.IGNORECASE)


def manifest_declared_substrate(roles: Any) -> dict[str, Any] | None:
    """Substrate the stack MANIFEST declares for a server's roles — not observed.

    RTG-47 data plane: an ``expected-stack-server`` node has no process
    evidence (nothing is running on its port), so ``_service_substrate`` /
    the binary-path rule cannot speak for it and the page used to fall back
    to a port/name heuristic. The manifest does declare where the role runs
    (``master_declared(role, "device")``), so read THAT — and label it as a
    declaration, never as an observation.

    Returns ``{"substrate": "gpu"|"cpu", "source": "manifest",
    "declared_by": <where the declaration lives>}`` or ``None`` when the
    manifest cannot be consulted (import failure) or the roles resolve to no
    master row at all. ``declared_by`` names the master row + binding for an
    explicit device, or ``"device-omitted"`` for a host lane, so an operator
    can tell "declared ROCm0" from "declared nothing, host by convention".
    """
    raw_roles = [str(r) for r in roles if isinstance(r, str) and r] if isinstance(roles, list) else []
    if not raw_roles:
        return None
    # Master rows are keyed by canonical role; an instance label ("frontdoor.q0")
    # or numbered sibling ("embedder_3") resolves through its base role.
    role_list: list[str] = []
    for raw in raw_roles:
        for cand in (raw, re.sub(r"\.q\d+$", "", raw), base_role(re.sub(r"\.q\d+$", "", raw))):
            if cand and cand not in role_list:
                role_list.append(cand)
    try:
        from scripts.server.stack_manifest import master_declared, master_server_row
    except Exception as exc:  # manifest unreadable: declare nothing, never guess
        logger.debug("stack manifest unavailable for declared substrate: %s", exc)
        return None
    for role in role_list:
        try:
            device, source = master_declared(role, "device")
        except Exception:
            continue
        if isinstance(device, str) and device.strip():
            substrate = "gpu" if _MANIFEST_GPU_DEVICE_RE.search(device) else "cpu"
            return {
                "substrate": substrate,
                "source": "manifest",
                "declared_by": f"{source or role} device={device}",
            }
    # No role declares a device. If at least one role HAS a master row the
    # manifest is declaring a host lane by omission; if none does, the
    # manifest knows nothing about these roles and we say nothing.
    for role in role_list:
        try:
            _name, row, _binding = master_server_row(role)
        except Exception:
            continue
        if row is not None:
            return {"substrate": "cpu", "source": "manifest", "declared_by": "device-omitted"}
    return None


def _service_substrate(pid: Any, model_hint: str = "") -> str | None:
    """Best-effort substrate for an aux service: read argv[0] from /proc.

    A HIP/ROCm marker in the binary path (or, failing that, an explicit marker
    in the state file's model label, e.g. ``whisper.cpp large-v3-turbo (HIP)``)
    reads as GPU. Anything else returns ``None`` — an arbitrary service binary
    without a marker proves nothing, and the dashboard renders unknown as
    unmarked rather than guessing.
    """
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
            argv0 = f.read().split(b"\0", 1)[0].decode("utf-8", "replace")
        if argv0 and _is_gpu_binary(argv0):
            return "gpu"
    except (OSError, ValueError, TypeError):
        pass
    # A GPU process must map the HIP/ROCm runtime, so /proc/<pid>/maps is
    # conclusive FOR gpu (catches sd-server, whose binary path carries no
    # marker). Its absence proves nothing about the SERVICE — python wrappers
    # (document_formalizer) do the inference in a child — so no-marker never
    # claims cpu here.
    try:
        with open(f"/proc/{int(pid)}/maps", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "amdhip" in line or "rocblas" in line or "libhsa" in line:
                    return "gpu"
    except (OSError, ValueError, TypeError):
        pass
    if model_hint and _SUBSTRATE_MARKER_RE.search(model_hint):
        return "gpu"
    return None


def _discover_llama_processes() -> dict[int, dict[str, Any]]:
    """Scan for llama-server listeners → {port: {role, attribution, ...}}.

    Superset of `_discover_llama_ports`: carries the same port→role labels plus
    the M3 attribution verdict per listener (``attribution``, optional
    ``lane_hint`` naming the configured lane a demoted label came from,
    optional ``marker_stale``, plus ``pid``/``started_at``). Cheap (~5ms), runs
    once per snapshot poll.
    """
    out = _ps_llama_scan()
    now = time.time()
    line_re = re.compile(r"^\s*(\d+)\s+(\d+)\s+")
    pid_port_re = re.compile(r"--port\s+(\d+)")
    service_ports = set(_BASE_SERVICE_PORT_HINTS)
    markers = _llama_fleet_markers()
    contract_ports = _launch_contract_ports()
    procs: dict[int, dict[str, Any]] = {}
    for line in out.splitlines():
        if "llama-server" not in line:
            continue
        port_m = pid_port_re.search(line)
        if not port_m:
            continue
        port = int(port_m.group(1))
        pid: int | None = None
        proc_started_at: float | None = None
        line_m = line_re.match(line)
        if line_m:
            pid = int(line_m.group(1))
            proc_started_at = now - float(line_m.group(2))
        verdict = _classify_llama_attribution(port, proc_started_at, markers, contract_ports)
        info: dict[str, Any] = {
            "pid": pid,
            "started_at": proc_started_at,
            "attribution": verdict["attribution"],
        }
        # For llama-server the binary IS the inference kernel and this host's
        # convention is established (CPU tree `build/`, GPU tree `build-hip/`),
        # so no-marker legitimately reads as CPU here — unlike aux services.
        binary = next((tok for tok in line.split() if "llama-server" in tok), "")
        if binary:
            info["substrate"] = "gpu" if _is_gpu_binary(binary) else "cpu"
        if verdict.get("marker_stale"):
            info["marker_stale"] = True
        role = _port_hint(port)
        if role == f"port_{port}":
            # Unmapped port: name it by what it IS. Mangling the model stem
            # into the role (`port_8802(Qwen3.6-…)`) leaked garbled keys into
            # live_busy_by_role and made the activity view unreadable.
            # MI210 HIP builds are the GPU testbed — operator-decided
            # (2026-07-05) to render first-class ahead of stack integration.
            # A FRESH fleet marker may still name the roles (a stack launch on
            # a port absent from the static hints); a stale marker never does.
            marker_roles = (markers.get(port) or {}).get("roles") or []
            if verdict["attribution"] == ATTRIBUTION_FLEET_MARKER and marker_roles:
                role = str(marker_roles[0])
            else:
                role = "mi210_gpu" if "mi210" in line.lower() else f"extern_{port}"
        elif port in service_ports:
            # Auxiliary service hints (orchestrator, embedders, sd, whisper)
            # are not production lanes; M3 demotion is scoped to lane labels.
            if verdict["attribution"] in (ATTRIBUTION_UNMANAGED, ATTRIBUTION_UNVERIFIED):
                info["attribution"] = ATTRIBUTION_SERVICE_HINT
                info.pop("marker_stale", None)
        elif verdict["attribution"] == ATTRIBUTION_UNMANAGED and base_role(role).startswith(
            "embedder"
        ):
            # Embedder siblings (8096-8098) sit outside _BASE_SERVICE_PORT_HINTS
            # but are auxiliary services all the same — never lane-demoted.
            info["attribution"] = ATTRIBUTION_SERVICE_HINT
            info.pop("marker_stale", None)
        elif verdict["attribution"] == ATTRIBUTION_UNMANAGED:
            # Lane-labeled listener with NO vouching evidence while the
            # attribution plane is live: the label came from a static lineup
            # or a stale marker — the M3 bug class. Demote to extern_<port>
            # and surface the lane it would have (mis)rendered under.
            info["lane_hint"] = role
            role = "mi210_gpu" if "mi210" in line.lower() else f"extern_{port}"
        info["role"] = role
        procs[port] = info
    return procs


def _discover_llama_ports() -> dict[int, str]:
    """Scan /proc for running llama-server processes and extract port→role.

    Falls back to _PORT_HINTS for unmapped ports; lane labels are demoted to
    ``extern_<port>`` when the listener has no fleet marker and no
    launch-contract membership (M3 — see `_discover_llama_processes`). Cheap
    (~5ms), runs once per snapshot poll.
    """
    return {port: info["role"] for port, info in _discover_llama_processes().items()}


_VENDOR_PREFIX_RE = re.compile(
    r"^(Qwen|Meta|Google|Mistral|DeepSeek|unsloth|bartowski|lmstudio[-_]community)[-_]",
    re.IGNORECASE,
)
_SHARD_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}$")


def _clean_model_name(model_path: str) -> str:
    """Human-friendly model label from a GGUF path.

    basename → drop `.gguf` → drop multi-file shard suffix (`-00001-of-00003`)
    → drop a redundant leading vendor prefix (`Qwen_Qwen3.6…` → `Qwen3.6…`).
    Returns '' for empty input so callers can omit the field.
    """
    if not model_path:
        return ""
    stem = Path(model_path).name
    stem = re.sub(r"\.gguf$", "", stem, flags=re.IGNORECASE)
    stem = _SHARD_SUFFIX_RE.sub("", stem)
    stem = _VENDOR_PREFIX_RE.sub("", stem)
    return stem


def _discover_llama_models() -> dict[int, str]:
    """Scan /proc for running llama-server processes → {port: cleaned model name}.

    Mirrors `_discover_llama_ports` but extracts the `-m <model>` GGUF path so the
    topology endpoint can label each role with the model it is actually serving.
    """
    models: dict[int, str] = {}
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,cmd"], capture_output=True, text=True, timeout=2,
        ).stdout
    except Exception:
        out = ""
    pid_port_re = re.compile(r"--port\s+(\d+)")
    pid_model_re = re.compile(r"-m\s+(\S+)")
    for line in out.splitlines():
        if "llama-server" not in line:
            continue
        port_m = pid_port_re.search(line)
        model_m = pid_model_re.search(line)
        if not port_m or not model_m:
            continue
        models[int(port_m.group(1))] = _clean_model_name(model_m.group(1))
    return models


def _pid_is_running(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    return pid_int > 0 and Path(f"/proc/{pid_int}").exists()


def _load_state_services(state_path: Path) -> list[dict[str, Any]]:
    """Load non-llama auxiliary services from orchestrator_state.json at `state_path`."""
    services: list[dict[str, Any]] = []
    try:
        with open(state_path) as f:
            state = json.load(f)
        for key, info in state.items():
            if not isinstance(info, dict):
                continue
            entry = {
                "name": key,
                "role": info.get("role", key),
                "port": info.get("port"),
                "model": info.get("model_path", ""),
                "pid": info.get("pid", -1),
                "running": _pid_is_running(info.get("pid")),
            }
            substrate = _service_substrate(info.get("pid"), str(info.get("model_path", "")))
            if substrate:
                entry["substrate"] = substrate
            services.append(entry)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.debug("Failed to load orchestrator_state.json: %s", exc)
    return services


def expected_stack_services(numa_mode: str | None = None) -> list[dict[str, Any]]:
    """Expected stack servers from the launch manifest, including unloaded ports."""
    if numa_mode is None and "ORCHESTRATOR_STACK_NUMA_MODE" not in os.environ:
        runtime_servers = read_runtime_stack_selected_servers()
        if runtime_servers is not None:
            return _expected_services_from_manifest_servers(runtime_servers)

    try:
        from scripts.server.stack_manifest import HOT_SERVERS, WARM_SERVERS, _filter_by_numa_mode
    except Exception as exc:
        # Was logger.debug — i.e. invisible in normal operation. Returning [] means
        # "no expected services", which renders as a healthy empty stack rather than
        # as a read failure, so the quiet level made an absence indistinguishable
        # from a fact.
        logger.warning(
            "stack manifest unavailable — reporting NO expected services (this is a "
            "READ FAILURE, not an empty stack): %s", exc,
        )
        return []

    mode = numa_mode or active_stack_numa_mode()
    try:
        servers = _filter_by_numa_mode(HOT_SERVERS + WARM_SERVERS, mode)
    except Exception as exc:
        # FAIL-OPEN, and deliberately kept: dropping the panel is worse than
        # over-reporting it. But it was logged at DEBUG, which is what made it a
        # MASK rather than a fallback — a filter bug produced a silently unfiltered
        # list that looks exactly like a correct one.
        #
        # The original justification was the scripts.server circular-import flake
        # (filed 2026-07-24). Re-derived 2026-08-12: that cycle NO LONGER
        # REPRODUCES — 3 bare and 4 package-order imports of
        # runtime_facts_manifest / stack_paths / stack_manifest / orchestrator_stack
        # all import cleanly. So anything caught here now is a REAL filter defect,
        # which is exactly the case that must not be quiet.
        logger.warning(
            "NUMA-mode filter FAILED for mode %r — falling back to the UNFILTERED "
            "manifest (%d servers). Ports for other modes will be reported as "
            "expected; this is a filter defect, not a topology change: %s",
            mode, len(HOT_SERVERS) + len(WARM_SERVERS), exc,
        )
        servers = HOT_SERVERS + WARM_SERVERS
    return _expected_services_from_manifest_servers(servers)


def _expected_services_from_manifest_servers(servers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    services: list[dict[str, Any]] = []
    for server in servers:
        if not isinstance(server, dict):
            continue
        port = server.get("port")
        roles = server.get("roles") or []
        if not isinstance(port, int) or not isinstance(roles, list) or not roles:
            continue
        role = _manifest_server_label(server) or _port_hint(port)
        if role == f"port_{port}":
            role = str(roles[0])
        services.append({
            "name": role,
            "role": role,
            "port": port,
            "roles": [str(r) for r in roles],
            "embedding": bool(server.get("embedding")),
            "vision": bool(server.get("vision")),
            "worker_pool": bool(server.get("worker_pool")),
            "numa_instance": server.get("numa_instance"),
        })
    return services


def _process_info_by_match(needle: str) -> dict[str, Any]:
    """Find a long-running Python process by command-line substring."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,etime,pcpu,cmd"],
            capture_output=True, text=True, timeout=2,
        ).stdout
    except Exception:
        return {"running": False}
    for line in out.splitlines()[1:]:
        if needle in line and "grep" not in line:
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            return {
                "running": True,
                "pid": int(parts[0]),
                "etime": parts[1],
                "pcpu": float(parts[2]),
                "cmd": parts[3][:200],
            }
    return {"running": False}
