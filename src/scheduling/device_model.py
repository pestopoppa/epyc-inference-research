"""DEVICE dimension for the placement-feasibility model.

Artifact 1 + 2 of `handoffs/active/contention-model-device-and-load-axes-rider.md`.

Why this module exists
──────────────────────
Before this, the contention model's entire resource vocabulary was "CPU
regions derived from NUMA cpusets" (`src/runtime/instance_topology.py`).
`src/scheduling/contention.py`, `src/scheduling/contention_gate.py` and
`src/runtime/cpu_region_lock.py` contained ZERO occurrences of `device`, `gpu`
or `ROCm`. So a role whose weights are VRAM-resident under `-ngl` — and whose
cpuset exists only to give it host threads for tokenising and sampling — was
accounted IDENTICALLY to a CPU decode holding the same cpuset.

Two wrong answers followed:

  1. FALSE EXCLUSION. A GPU lane overlapping a half/full CPU instance was
     called a conflict, although its real draw on the contended resource
     (DRAM bandwidth over those regions) is close to nil.
  2. UNMODELLED CONTENTION. Two GPU roles sharing NO cpuset still contend for
     VRAM capacity and HBM bandwidth, and the model could not see it. The
     measured four-model steady state was 62.59 of 63.98 GiB — 1.40 GiB
     headroom — and VRAM grows on first EXECUTION, not at load.

Derivation, not restatement
───────────────────────────
Everything here is DERIVED from declared artifacts. There is no per-role
device table in this file and no hardcoded VRAM figure:

  * device            ← `orchestration/derived/stack_priors.yaml`
                        `roles.<r>.serving.launch.runtime.flags.device`
                        (AUTHORITATIVE), corroborated by
                        `NUMA_CONFIG[<r>]["gpu_host_lane"]`.
  * per-role VRAM     ← the same compiled priors
                        (`roles.<r>.evidence.quality[*].value.vram_gib|vram_mb`),
                        falling back to `orchestration/model_registry.yaml`
                        `server_mode.<r>.vram_gib|vram_mb`.
  * device capacity   ← `orchestration/gpu_shadow_lane_np_ceiling.yaml`
                        `vram_total_gib` (device-stamped `device: ROCm0`),
                        overridable by env, with an optional READ-ONLY
                        `rocm-smi --showmeminfo vram` host query.

DISAGREEMENT IS AN ERROR, NEVER A VOTE. If the compiled priors and
`gpu_host_lane` disagree about a role, `resolve_role_device` RAISES. Silently
picking one of them is the exact defect class this module exists to remove.
Likewise a role that neither source knows about RAISES — it is never defaulted
to CPU, because defaulting to CPU is what produced the bug.

This module is pure and import-safe: no processes are started, inspected or
signalled, and the only optional host access is a read-only `rocm-smi` query
that is off unless a caller explicitly opts in.
"""

from __future__ import annotations

import enum
import functools
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

log = logging.getLogger("scheduling.device_model")

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_STACK_PRIORS_PATH = _REPO_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
DEFAULT_REGISTRY_PATH = _REPO_ROOT / "orchestration" / "model_registry.yaml"
# Declared device capacity. `gpu_shadow_lane_np_ceiling.yaml` is the repo's only
# device-stamped VRAM-total declaration (`device: ROCm0`, `vram_total_gib: 64`);
# it is policy-as-data and is read here for the constant only.
DEFAULT_DEVICE_CAPACITY_PATH = (
    _REPO_ROOT / "orchestration" / "gpu_shadow_lane_np_ceiling.yaml"
)

# ── VRAM headroom reserve ────────────────────────────────────────────
#
# 2.0 GiB, default. Justification (all measured, none guessed):
#
#   * The declared total is the NOMINAL 64 GiB; `rocm-smi` reports 63.98 GiB
#     usable, so ~0.02 GiB is gone before any tenant loads.
#   * The declared per-role figures are LOAD-TIME footprints. The measured
#     four-model steady state grew from 61.66 GiB at load to 62.59 GiB after
#     each model had executed once — +0.93 GiB of post-first-execution
#     growth that an admission-time sum over declared weights cannot see.
#     ("VRAM grows on first EXECUTION, not at load" — the registry's own
#     `vram_caveat`, which warns that load-time budgets run ~1 GiB optimistic.)
#   * 2.0 GiB covers that measured 0.93 GiB growth with ~2x margin and still
#     leaves the live GPU set (architect_general 36.70 + worker_vision 20.56
#     = 57.26 GiB) feasible with ~4.7 GiB to spare. (The 36.70 GiB 27B process
#     serves architect_critic since the 2026-09-27 ARCHITECT SWAP.)
#
# This is a HARD constraint that fails CLOSED (rider §4), so erring large is
# the correct direction. Override with `ORCHESTRATOR_VRAM_HEADROOM_GIB`.
DEFAULT_VRAM_HEADROOM_GIB = 2.0
VRAM_HEADROOM_ENV = "ORCHESTRATOR_VRAM_HEADROOM_GIB"
VRAM_TOTAL_ENV = "ORCHESTRATOR_VRAM_TOTAL_GIB"

# Device-token vocabulary. RESTATED, not derived: the tokens are llama.cpp
# `--device` backend names, which live in the kernel tree, not in any
# orchestrator artifact. Matching is prefix-based and case-insensitive so
# "ROCm0"/"ROCm1"/"CUDA0" all classify without enumerating card indices.
_GPU_DEVICE_PREFIXES = ("rocm", "cuda", "gpu", "hip", "vulkan", "sycl", "metal")
# Tokens that positively declare "no accelerator". `None`/absent is handled
# separately from these — see `declared_device`.
_CPU_DEVICE_TOKENS = frozenset({"none", "null", "cpu", "blas", ""})

GIB_PER_MIB = 1.0 / 1024.0


class DeviceClass(str, enum.Enum):
    """Which resource an instance's exclusivity claim is actually about."""

    CPU = "cpu"
    GPU = "gpu"


class DeviceResolutionError(RuntimeError):
    """Device could not be resolved, or the declared sources disagree.

    Raised — never swallowed into a default — because a silent disagreement
    between the compiled priors and `gpu_host_lane` is precisely the defect
    class the device axis was added to eliminate.
    """


@dataclass(frozen=True)
class RoleDevice:
    """Resolved device facts for one role."""

    role: str
    device_class: DeviceClass
    device: str | None  # raw declared token, e.g. "ROCm0"; None for CPU roles
    source: str  # "stack_priors" | "numa_config"
    corroborated: bool  # both sources present AND agreed
    # Alias-collapsing key. `vision_escalation` and `worker_vision` are two
    # roles on ONE :8086 process; summing both would double-count ~20.6 GiB.
    # `serving.server_role` in the compiled priors is the declared collapse.
    accounting_key: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.device_class is DeviceClass.GPU


@dataclass(frozen=True)
class RoleVram:
    """Declared VRAM footprint for one role."""

    role: str
    gib: float | None
    source: str  # "stack_priors" | "model_registry" | "" when undeclared
    accounting_key: str = ""


@dataclass(frozen=True)
class VramFit:
    """Result of the capacity check over the GPU subset of a role-set."""

    ok: bool
    reason: str  # "" | "vram_capacity_exceeded" | "vram_declaration_missing"
    required_gib: float
    budget_gib: float
    capacity_gib: float
    headroom_gib: float
    capacity_source: str
    per_role: dict[str, float] = field(default_factory=dict)
    undeclared_roles: tuple[str, ...] = ()
    counted_roles: tuple[str, ...] = ()

    @property
    def slack_gib(self) -> float:
        return round(self.budget_gib - self.required_gib, 4)


# ────────────────────────────────────────────────────────────────────
# Artifact loading (memoized by path + mtime so tests can point elsewhere)
# ────────────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        import yaml
    except Exception:  # noqa: BLE001 — keep import-safe
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, Exception):  # noqa: BLE001 — YAML errors vary by loader
        return None
    return data if isinstance(data, dict) else None


@functools.lru_cache(maxsize=8)
def _load_cached(path_str: str, mtime: float) -> dict[str, Any] | None:
    return _load_yaml(Path(path_str))


def _load_artifact(path: Path | None, default: Path) -> dict[str, Any] | None:
    p = Path(path) if path is not None else default
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    return _load_cached(str(p), mtime)


def load_priors(path: Path | None = None) -> dict[str, Any]:
    """`roles` block of the compiled stack priors, or {} when unavailable."""
    art = _load_artifact(path, DEFAULT_STACK_PRIORS_PATH)
    roles = (art or {}).get("roles")
    return roles if isinstance(roles, dict) else {}


def load_registry_server_mode(path: Path | None = None) -> dict[str, Any]:
    """`server_mode` block of the model registry, or {} when unavailable."""
    art = _load_artifact(path, DEFAULT_REGISTRY_PATH)
    sm = (art or {}).get("server_mode")
    return sm if isinstance(sm, dict) else {}


def _live_numa_config() -> dict[str, Any]:
    """Read NUMA_CONFIG lazily. Returns {} if the import path differs."""
    try:
        from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]

        return dict(NUMA_CONFIG)
    except Exception:  # noqa: BLE001 — import-safety over convenience
        return {}


# ────────────────────────────────────────────────────────────────────
# Device resolution
# ────────────────────────────────────────────────────────────────────


def classify_device_token(token: Any) -> DeviceClass | None:
    """Classify a raw `--device` token.

    Returns GPU for an accelerator token, CPU for an explicit no-accelerator
    token (including a declared `null`, which is how the compiled priors say
    "CPU"), and None only for a token this vocabulary does not recognise —
    which callers must treat as unresolved, never as CPU.
    """
    if token is None:
        return DeviceClass.CPU
    text = str(token).strip().lower()
    if text in _CPU_DEVICE_TOKENS:
        return DeviceClass.CPU
    if text.startswith(_GPU_DEVICE_PREFIXES):
        return DeviceClass.GPU
    return None


def declared_device(
    role: str, priors: Mapping[str, Any] | None = None
) -> tuple[bool, Any]:
    """AUTHORITATIVE device declaration for `role` from the compiled priors.

    Returns `(present, token)`. `present` is False when the priors carry no
    record for the role or no `runtime.flags` block at all — distinct from
    `present=True, token=None`, which is a POSITIVE declaration of "no
    accelerator" and classifies as CPU.
    """
    roles = priors if priors is not None else load_priors()
    record = roles.get(role) if isinstance(roles, Mapping) else None
    if not isinstance(record, Mapping):
        return (False, None)
    flags = (
        record.get("serving", {})
        .get("launch", {})
        .get("runtime", {})
        .get("flags", {})
        if isinstance(record.get("serving"), Mapping)
        else None
    )
    if not isinstance(flags, Mapping) or "device" not in flags:
        return (False, None)
    return (True, flags.get("device"))


def gpu_host_lane_flag(
    role: str, numa_config: Mapping[str, Any] | None = None
) -> bool | None:
    """Corroborating signal from NUMA_CONFIG.

    Returns True/False when the role IS in NUMA_CONFIG — within that config's
    own grammar an absent `gpu_host_lane` key means False, which is exactly how
    `stack_numa._assert_instance_invariants` already reads it — and None when
    the role is absent from NUMA_CONFIG entirely (no signal at all).
    """
    cfg = numa_config if numa_config is not None else _live_numa_config()
    record = cfg.get(role) if isinstance(cfg, Mapping) else None
    if not isinstance(record, Mapping):
        return None
    return bool(record.get("gpu_host_lane"))


def _accounting_key(role: str, priors: Mapping[str, Any]) -> str:
    record = priors.get(role) if isinstance(priors, Mapping) else None
    if isinstance(record, Mapping):
        serving = record.get("serving")
        if isinstance(serving, Mapping):
            server_role = serving.get("server_role")
            if isinstance(server_role, str) and server_role:
                return server_role
    return role


def resolve_role_device(
    role: str,
    *,
    numa_config: Mapping[str, Any] | None = None,
    priors: Mapping[str, Any] | None = None,
) -> RoleDevice:
    """Resolve `role`'s device class. RAISES rather than guessing.

    Rules, in order:
      1. Both sources present and DISAGREE  → `DeviceResolutionError`.
         (A `device: ROCm0` role without `gpu_host_lane`, or a `gpu_host_lane`
         role whose priors declare no accelerator, is a corrupted topology
         declaration — not a tie to be broken.)
      2. Both present and agree             → resolved, `corroborated=True`.
      3. Only the priors declare            → resolved from the priors.
      4. Only NUMA_CONFIG knows the role    → resolved from `gpu_host_lane`,
         the explicitly-permitted fallback signal.
      5. Neither source knows the role, or the priors carry a device token this
         vocabulary cannot classify → `DeviceResolutionError`. NEVER CPU.
    """
    priors_map = priors if priors is not None else load_priors()
    present, token = declared_device(role, priors_map)
    lane = gpu_host_lane_flag(role, numa_config)

    prior_class: DeviceClass | None = None
    if present:
        prior_class = classify_device_token(token)
        if prior_class is None:
            raise DeviceResolutionError(
                f"role {role!r}: stack_priors declares device {token!r}, which is "
                f"not a recognised device token. Refusing to guess a device class; "
                f"add the backend to _GPU_DEVICE_PREFIXES or fix the declaration."
            )

    lane_class: DeviceClass | None = None
    if lane is not None:
        lane_class = DeviceClass.GPU if lane else DeviceClass.CPU

    key = _accounting_key(role, priors_map)

    if prior_class is not None and lane_class is not None:
        if prior_class is not lane_class:
            raise DeviceResolutionError(
                f"role {role!r}: DEVICE DECLARATIONS DISAGREE — "
                f"stack_priors roles.{role}.serving.launch.runtime.flags.device="
                f"{token!r} implies {prior_class.value}, but "
                f"NUMA_CONFIG[{role!r}]['gpu_host_lane']={lane!r} implies "
                f"{lane_class.value}. Refusing to pick one: a silent disagreement "
                f"here is the exact defect the device axis exists to eliminate. "
                f"Fix the declaration in whichever artifact is wrong."
            )
        return RoleDevice(
            role=role,
            device_class=prior_class,
            device=token if prior_class is DeviceClass.GPU else None,
            source="stack_priors",
            corroborated=True,
            accounting_key=key,
        )

    if prior_class is not None:
        return RoleDevice(
            role=role,
            device_class=prior_class,
            device=token if prior_class is DeviceClass.GPU else None,
            source="stack_priors",
            corroborated=False,
            accounting_key=key,
        )

    if lane_class is not None:
        return RoleDevice(
            role=role,
            device_class=lane_class,
            device=None,
            source="numa_config",
            corroborated=False,
            accounting_key=key,
        )

    raise DeviceResolutionError(
        f"role {role!r}: no device declaration in stack_priors "
        f"(roles.{role}.serving.launch.runtime.flags.device) and no NUMA_CONFIG "
        f"entry to corroborate it. Refusing to default to CPU — defaulting to CPU "
        f"is what produced the false-exclusion bug."
    )


def resolve_device_classes(
    roles: Iterable[str],
    *,
    numa_config: Mapping[str, Any] | None = None,
    priors: Mapping[str, Any] | None = None,
) -> dict[str, RoleDevice]:
    """Resolve a whole role-set at once. Raises on the FIRST unresolvable role
    so the failure surfaces before any feasibility answer is produced."""
    priors_map = priors if priors is not None else load_priors()
    cfg = numa_config if numa_config is not None else _live_numa_config()
    return {
        role: resolve_role_device(role, numa_config=cfg, priors=priors_map)
        for role in roles
    }


# ────────────────────────────────────────────────────────────────────
# VRAM: per-role footprint + device capacity
# ────────────────────────────────────────────────────────────────────


def _vram_from_value(value: Any) -> float | None:
    """Pull a VRAM figure out of one declaration block (direct keys only).

    Direct keys only, deliberately: role records nest
    `superseded_model_history` blocks describing models that are no longer
    deployed, and a recursive scan would happily bill their footprint.
    """
    if not isinstance(value, Mapping):
        return None
    if isinstance(value.get("vram_gib"), (int, float)):
        return float(value["vram_gib"])
    if isinstance(value.get("vram_mb"), (int, float)):
        return float(value["vram_mb"]) * GIB_PER_MIB
    return None


def _resolve_server_mode(
    priors: Mapping[str, Any] | None, server_mode: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    """Pick the registry fallback WITHOUT leaking the live artifact into a
    caller-supplied closed world.

    A caller that passes explicit `priors` is describing a synthetic or pinned
    universe. Quietly reaching past it into the on-disk registry would let a
    role that declares no VRAM in the supplied priors still resolve — the
    fail-open-fallback pattern that conceals its own gaps. So: explicit
    `server_mode` wins; otherwise the live registry is read ONLY when `priors`
    was also left to the live artifact.
    """
    if server_mode is not None:
        return server_mode
    if priors is not None:
        return {}
    return load_registry_server_mode()


def declared_vram_gib(
    role: str,
    *,
    priors: Mapping[str, Any] | None = None,
    server_mode: Mapping[str, Any] | None = None,
) -> RoleVram:
    """Declared VRAM for `role`, in GiB.

    Priors first (`evidence.quality[*].value`), registry `server_mode` as the
    fallback. When several declarations exist the LARGEST is taken: this feeds
    a hard, fail-closed capacity gate, so over-stating is the safe direction
    and under-stating is not.
    """
    priors_map = priors if priors is not None else load_priors()
    key = _accounting_key(role, priors_map)

    candidates: list[float] = []
    record = priors_map.get(role) if isinstance(priors_map, Mapping) else None
    if isinstance(record, Mapping):
        evidence = record.get("evidence")
        if isinstance(evidence, Mapping):
            for block in evidence.values():
                entries = block if isinstance(block, list) else [block]
                for entry in entries:
                    if not isinstance(entry, Mapping):
                        continue
                    got = _vram_from_value(entry.get("value"))
                    if got is not None:
                        candidates.append(got)
        # `performance`-style blocks, if a future compile lifts them out of
        # `evidence`, are read the same way rather than being re-specified.
        for direct_key in ("performance", "priors", "server_mode"):
            got = _vram_from_value(record.get(direct_key))
            if got is not None:
                candidates.append(got)
    if candidates:
        return RoleVram(role=role, gib=round(max(candidates), 4),
                        source="stack_priors", accounting_key=key)

    sm = _resolve_server_mode(priors, server_mode)
    got = _vram_from_value(sm.get(role)) if isinstance(sm, Mapping) else None
    if got is not None:
        return RoleVram(role=role, gib=round(got, 4),
                        source="model_registry", accounting_key=key)

    return RoleVram(role=role, gib=None, source="", accounting_key=key)


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw.strip())
    except ValueError:
        log.warning("%s=%r is not a number — ignoring", name, raw)
        return None


def vram_headroom_gib(override: float | None = None) -> float:
    """Configurable reserve. Explicit argument > env > module default."""
    if override is not None:
        return float(override)
    env = _env_float(VRAM_HEADROOM_ENV)
    return env if env is not None else DEFAULT_VRAM_HEADROOM_GIB


def _rocm_smi_total_gib() -> float | None:
    """READ-ONLY `rocm-smi --showmeminfo vram` query for card 0.

    Off unless a caller opts in. Starts no process on the serving path, kills
    nothing, and changes no device state — it only reads the meminfo table.
    """
    try:
        proc = subprocess.run(
            ["rocm-smi", "-d", "0", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001 — absent binary, timeout, perms
        log.warning("rocm-smi VRAM query unavailable: %s", exc)
        return None
    if proc.returncode != 0:
        log.warning("rocm-smi exited %s", proc.returncode)
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    for card in (payload or {}).values():
        if not isinstance(card, Mapping):
            continue
        for key, value in card.items():
            if "vram total memory" in str(key).lower():
                try:
                    return float(value) / (1024.0**3)
                except (TypeError, ValueError):
                    continue
    return None


def vram_capacity_gib(
    *,
    capacity_path: Path | None = None,
    allow_host_query: bool = False,
) -> tuple[float, str]:
    """Total device VRAM in GiB, plus the source it came from.

    Order: explicit env override → declared artifact → optional read-only
    `rocm-smi` → raise. Never a hardcoded number.
    """
    env = _env_float(VRAM_TOTAL_ENV)
    if env is not None:
        return (env, f"env:{VRAM_TOTAL_ENV}")

    art = _load_artifact(capacity_path, DEFAULT_DEVICE_CAPACITY_PATH)
    if isinstance(art, Mapping):
        total = art.get("vram_total_gib")
        if isinstance(total, (int, float)):
            src = str(
                Path(capacity_path) if capacity_path else DEFAULT_DEVICE_CAPACITY_PATH
            )
            return (float(total), f"declared:{src}")

    if allow_host_query:
        got = _rocm_smi_total_gib()
        if got is not None:
            return (got, "rocm-smi")

    raise DeviceResolutionError(
        "device VRAM capacity is undeclared: no "
        f"{VRAM_TOTAL_ENV} override, no `vram_total_gib` in "
        f"{capacity_path or DEFAULT_DEVICE_CAPACITY_PATH}"
        + ("" if allow_host_query else " (host query not permitted)")
        + ". Refusing to assume a capacity."
    )


def vram_fit(
    roles: Iterable[str],
    *,
    priors: Mapping[str, Any] | None = None,
    server_mode: Mapping[str, Any] | None = None,
    headroom_gib: float | None = None,
    capacity_gib: float | None = None,
    capacity_path: Path | None = None,
    allow_host_query: bool = False,
) -> VramFit:
    """Does this set of GPU roles fit in the device, with headroom reserved?

    `roles` must already be the GPU subset — callers partition by device
    first. Aliases that share one server process (`serving.server_role`) are
    counted ONCE.

    Fails CLOSED (rider §4: contention management owns hard constraints):
      * any GPU role with no declared VRAM → `vram_declaration_missing`
      * declared sum over budget            → `vram_capacity_exceeded`
    """
    sm = _resolve_server_mode(priors, server_mode)
    priors_map = priors if priors is not None else load_priors()

    if capacity_gib is not None:
        capacity, capacity_source = float(capacity_gib), "explicit"
    else:
        capacity, capacity_source = vram_capacity_gib(
            capacity_path=capacity_path, allow_host_query=allow_host_query
        )
    headroom = vram_headroom_gib(headroom_gib)
    budget = round(capacity - headroom, 4)

    per_role: dict[str, float] = {}
    undeclared: list[str] = []
    by_key: dict[str, float] = {}
    for role in sorted(set(roles)):
        rv = declared_vram_gib(role, priors=priors_map, server_mode=sm)
        if rv.gib is None:
            undeclared.append(role)
            continue
        per_role[role] = rv.gib
        # Alias collapse: two roles on one process are one VRAM footprint.
        by_key[rv.accounting_key] = max(by_key.get(rv.accounting_key, 0.0), rv.gib)

    required = round(sum(by_key.values()), 4)
    counted = tuple(sorted(per_role))

    if undeclared:
        return VramFit(
            ok=False,
            reason="vram_declaration_missing",
            required_gib=required,
            budget_gib=budget,
            capacity_gib=capacity,
            headroom_gib=headroom,
            capacity_source=capacity_source,
            per_role=per_role,
            undeclared_roles=tuple(sorted(undeclared)),
            counted_roles=counted,
        )

    return VramFit(
        ok=required <= budget,
        reason="" if required <= budget else "vram_capacity_exceeded",
        required_gib=required,
        budget_gib=budget,
        capacity_gib=capacity,
        headroom_gib=headroom,
        capacity_source=capacity_source,
        per_role=per_role,
        undeclared_roles=(),
        counted_roles=counted,
    )


def clear_caches() -> None:
    """Drop memoized artifact reads (tests that rewrite artifacts in place)."""
    _load_cached.cache_clear()
